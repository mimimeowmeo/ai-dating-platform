// 背景 AI 工作的生產者與結果消費者（規格 8）。
//
// 流程：NestJS 把要處理的資料放進 BullMQ 的 ai-jobs → Python AI worker 只做運算 →
// 結果放進 ai-results → 這裡取出來寫進資料庫（ADR 0002：只有 NestJS 碰資料庫）。
//
// 四種工作：
// - chunk-embed：把新訊息切片並向量化，供 RAG 檢索。
// - topic-spans：找出 AI 推薦開啟的話題區段，萃取風格時要降權。
// - summarize：每 50 則更新一次聊天室摘要。
// - build-style：重新萃取某位使用者的風格卡。
import { Injectable, OnModuleDestroy, OnModuleInit } from "@nestjs/common";
import { Queue, Worker, type Job } from "bullmq";
import { Database, config } from "./core";
import { AiData, JOB_MESSAGES } from "./ai-data";
import { VectorStore } from "./ai-store";
import {
  chunkResult,
  styleProfileResult,
  summaryResult,
  topicSpanResult,
  type AiChunkRequest,
  type AiStyleProfileRequest,
  type AiSummaryRequest,
  type AiTopicSpanRequest,
} from "./ai-client";

/** 送工作給 Python AI worker 的佇列。 */
export const JOBS_QUEUE = "ai-jobs";
/** AI worker 回傳結果的佇列；由這個程序消費。 */
export const RESULTS_QUEUE = "ai-results";
/** 切片與話題區段的 debounce：最後一則訊息之後 2 分鐘才處理，連續聊天只會跑一次。 */
const CHUNK_DEBOUNCE_MS = 2 * 60 * 1000;
/** 每累積這麼多則新訊息就更新一次摘要（規格 5.5）。 */
const SUMMARY_EVERY = 50;
/** 摘要上限字數。 */
const SUMMARY_MAX_CHARS = 600;
/** 使用者新增這麼多則真人訊息就重新萃取風格卡（規格 8）。 */
const STYLE_EVERY = 200;
/** 同一位使用者的風格卡工作 1 小時內只排一次，避免熱門使用者一直重算。 */
const STYLE_THROTTLE_MS = 60 * 60 * 1000;
/**
 * 送出的工作設定：失敗時重試 3 次（指數退避），成功就刪除。
 * 失敗的保留 50 筆方便查問題；job 內容含聊天訊息，不無限期留在 Redis。
 */
const JOB_OPTIONS = {
  attempts: 3,
  backoff: { type: "exponential" as const, delay: 5000 },
  removeOnComplete: true,
  removeOnFail: 50,
};

@Injectable()
export class AiJobs implements OnModuleInit, OnModuleDestroy {
  /**
   * BullMQ 要求連線的 maxRetriesPerRequest 是 null（它自己處理重試），
   * 所以不能共用 Infrastructure 的 Redis 連線，這裡自己開一組設定。
   */
  private readonly connection = {
    url: config.REDIS_URL,
    maxRetriesPerRequest: null,
  };
  private readonly queue = new Queue(JOBS_QUEUE, {
    connection: this.connection,
  });
  private results?: Worker;

  constructor(
    private db: Database,
    private data: AiData,
    private vectors: VectorStore,
  ) {}

  /** 服務啟動時開始消費結果佇列；concurrency 2 就夠，寫入都是小交易。 */
  onModuleInit() {
    // Queue 與 Worker 都是 EventEmitter：沒有接 error 事件時，Redis 一斷線就會讓整個程序結束。
    // AI 是附加功能，Redis 有問題時其他功能要照常運作，所以只記錄固定字串。
    this.queue.on("error", () => console.error("ai_jobs_queue_error"));
    this.results = new Worker(
      RESULTS_QUEUE,
      (job) => this.handleResult(job),
      { connection: this.connection, concurrency: 2 },
    );
    // 只記錄固定字串：錯誤物件可能含 job 資料（聊天內容），不寫進日誌。
    this.results.on("error", () => console.error("ai_results_worker_error"));
    this.results.on("failed", (_job, error) =>
      console.error("ai_result_failed", error?.message ?? "unknown"),
    );
  }

  /** 關機時先關 worker 再關佇列，讓正在處理的結果寫完。 */
  async onModuleDestroy() {
    await this.results?.close();
    await this.queue.close();
  }

  /**
   * 送出一則訊息之後要做的背景工作（呼叫端不等待，失敗不影響送訊息）。
   *
   * - 切片與話題區段：每次都排，但用 debounce 壓成「停止聊天 2 分鐘後跑一次」。
   * - 摘要：距離上一份摘要滿 50 則才排。
   * - 風格卡：送訊息的人累積滿 200 則真人訊息才排。
   */
  async afterMessage(conversationId: string, senderId: string) {
    await this.enqueueChunking(conversationId);
    await this.enqueueSummary(conversationId);
    await this.enqueueStyleIfDue(senderId);
  }

  /**
   * 排切片與話題區段。
   *
   * 兩個工作用同一批訊息：從「還沒結束的片段」的第一則開始（規格 5.2 的增量更新），
   * 沒有任何片段時就從整個聊天室的開頭開始。
   * deduplication 的 extend+replace 是 debounce：同一個聊天室連續來訊息時，
   * 延遲會一直往後延，資料換成最新的一份，最後只跑一次。
   */
  private async enqueueChunking(conversationId: string) {
    const anchor = await this.vectors.openChunkStart(conversationId);
    const messages = await this.data.indexMessages(
      conversationId,
      anchor ? { at: anchor.at, inclusive: true } : null,
      JOB_MESSAGES,
    );
    if (!messages.length) return;
    const chunkJob: AiChunkRequest = {
      conversationId,
      messages,
      embed: true,
    };
    const topicJob: AiTopicSpanRequest = { conversationId, messages };
    await this.queue.add("chunk-embed", chunkJob, {
      ...JOB_OPTIONS,
      delay: CHUNK_DEBOUNCE_MS,
      deduplication: {
        id: `conv:${conversationId}`,
        ttl: CHUNK_DEBOUNCE_MS,
        extend: true,
        replace: true,
      },
    });
    await this.queue.add("topic-spans", topicJob, {
      ...JOB_OPTIONS,
      delay: CHUNK_DEBOUNCE_MS,
      deduplication: {
        id: `spans:${conversationId}`,
        ttl: CHUNK_DEBOUNCE_MS,
        extend: true,
        replace: true,
      },
    });
  }

  /**
   * 距離上一份摘要滿 50 則新訊息就重新摘要；job 只帶「上次涵蓋到的訊息之後」的內容，
   * 加上舊摘要本身，讓模型用「舊摘要＋新訊息」增量更新（規格 5.5）。
   */
  private async enqueueSummary(conversationId: string) {
    const previous = await this.db.conversationSummary.findFirst({
      where: { conversationId },
      orderBy: { createdAt: "desc" },
    });
    const until = previous
      ? await this.db.message.findUnique({
          where: { id: previous.untilMessageId },
          select: { createdAt: true },
        })
      : null;
    const pending = await this.db.message.count({
      where: {
        conversationId,
        ...(until ? { createdAt: { gt: until.createdAt } } : {}),
      },
    });
    if (pending < SUMMARY_EVERY) return;
    const messages = await this.data.indexMessages(
      conversationId,
      until ? { at: until.createdAt, inclusive: false } : null,
      JOB_MESSAGES,
    );
    if (!messages.length) return;
    const job: AiSummaryRequest = {
      conversationId,
      previousSummary: previous?.summary ?? null,
      messages,
      maxChars: SUMMARY_MAX_CHARS,
    };
    await this.queue.add("summarize", job, {
      ...JOB_OPTIONS,
      // 摘要不需要 debounce：滿 50 則才會排，同一個聊天室同時只留一個待辦。
      deduplication: { id: `sum:${conversationId}`, ttl: CHUNK_DEBOUNCE_MS },
    });
  }

  /** 使用者自上次萃取後又累積滿 200 則真人訊息，就重新萃取風格卡。 */
  private async enqueueStyleIfDue(userId: string) {
    const profile = await this.db.userStyleProfile.findFirst({
      where: { userId },
      orderBy: { version: "desc" },
      select: { windowTo: true },
    });
    const since = profile?.windowTo ?? null;
    const added = await this.db.message.count({
      where: {
        senderId: userId,
        OR: [{ origin: { is: null } }, { origin: { origin: "human" } }],
        ...(since ? { createdAt: { gt: since } } : {}),
      },
    });
    if (added < STYLE_EVERY) return;
    await this.enqueueStyle(userId);
  }

  /**
   * 排一次風格卡萃取。
   *
   * 產生推薦時發現對方沒有風格卡也會呼叫這裡（規格 8 的「需要時發現沒有」），
   * 所以用 1 小時的 deduplication 節流：期間內同一位使用者只會排一次。
   * 沒有任何真人訊息時不排：AI 服務只會用 bio 做冷啟動，那件事線上就能做，不用背景工作。
   */
  async enqueueStyle(userId: string) {
    const user = await this.db.user.findUnique({
      where: { id: userId },
      select: { profile: { select: { bio: true } } },
    });
    const window = await this.data.ownMessages(userId);
    if (!window.messages.length) return;
    const job: AiStyleProfileRequest = {
      userId,
      bio: user?.profile?.bio ?? "",
      messages: window.messages,
      windowFrom: window.windowFrom,
      windowTo: window.windowTo,
      capped: window.capped,
      siteStats: await this.data.siteStats(),
      embedFacets: true,
    };
    await this.queue.add("build-style", job, {
      ...JOB_OPTIONS,
      deduplication: { id: `style:${userId}`, ttl: STYLE_THROTTLE_MS },
    });
  }

  /**
   * 處理 AI worker 回傳的一個結果。
   *
   * job.data 的格式是 { sourceJobId, name, key, result }（規格 8）。
   * 結果先用 zod 驗證再寫入：worker 是另一個程序、另一種語言，不能假設格式一定對。
   * 驗證或寫入失敗就丟出例外，讓 BullMQ 依 worker 設定的次數重試。
   */
  private async handleResult(job: Job) {
    const payload = job.data as { name?: string; result?: unknown };
    const name = payload?.name ?? job.name;
    if (name === "chunk-embed") {
      await this.vectors.replaceChunks(chunkResult.parse(payload.result));
      return;
    }
    if (name === "topic-spans") {
      const result = topicSpanResult.parse(payload.result);
      for (const span of result.spans) {
        const row = {
          conversationId: result.conversationId,
          initiatingMessageId: span.initiatingMessageId,
          initiatorId: span.initiatorId,
          startMessageId: span.startMessageId,
          endMessageId: span.endMessageId,
          messageCount: span.messageCount,
          endReason: span.endReason,
          detectorVersion: result.detectorVersion,
        };
        // 同一則發起訊息重算時會得到新的終點（後面又聊了幾則），所以用 upsert 覆蓋。
        await this.db.aiTopicSpan.upsert({
          where: {
            conversationId_initiatingMessageId: {
              conversationId: result.conversationId,
              initiatingMessageId: span.initiatingMessageId,
            },
          },
          create: row,
          update: row,
        });
      }
      return;
    }
    if (name === "summarize") {
      const result = summaryResult.parse(payload.result);
      await this.db.conversationSummary.create({
        data: {
          conversationId: result.conversationId,
          untilMessageId: result.untilMessageId,
          summary: result.summary,
          modelName: result.modelName ?? null,
          promptVersion: result.promptVersion,
        },
      });
      return;
    }
    if (name === "build-style") {
      const result = styleProfileResult.parse(payload.result);
      const userId = result.card.userId ?? String(job.data?.key ?? "");
      if (!userId) throw new Error("STYLE_RESULT_WITHOUT_USER");
      await this.vectors.insertStyleProfile(userId, result);
      return;
    }
    throw new Error("UNSUPPORTED_RESULT_JOB");
  }
}
