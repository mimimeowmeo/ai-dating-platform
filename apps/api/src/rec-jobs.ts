// 外貌向量的背景工作：新的主照片 → rec-jobs（services/recommendation 的 worker 算向量）
// → rec-results → 這裡寫進 appearance_embeddings（ADR 0002：只有 NestJS 碰資料庫）。
//
// 佇列和 AI 服務的 ai-jobs／ai-results 分開：兩邊的 worker 遇到不認得的 job 都會直接失敗。
// 照片原檔放進 job（上傳時已轉成長邊 1600px 以內的 JPEG），不再縮圖或重新壓縮：ml/ 離線算種子向量時
// 讀的也是原檔，兩邊的向量才一致（重新壓縮會讓少數照片的去背範圍改變）。worker 取到 job 就把照片從 Redis 清掉。
// worker 沒在跑時不排（照片不會堆在 Redis），等它上線後由每分鐘的補排接手。
import { Injectable, OnModuleDestroy, OnModuleInit } from "@nestjs/common";
import { Queue, Worker, type Job } from "bullmq";
import { z } from "zod";
import { Database, Infrastructure, config } from "./core";
import { APPEARANCE_DIMENSIONS, VectorStore } from "./ai-store";
import { projectOut } from "./discovery-rank";

export const REC_JOBS_QUEUE = "rec-jobs";
export const REC_RESULTS_QUEUE = "rec-results";
export const EMBED_APPEARANCE = "embed-appearance";
/**
 * 推薦 worker 的在線訊號：worker 每 20 秒寫一次、60 秒過期，正常關閉時刪掉（services/recommendation/app/worker.py）。
 * 不用 BullMQ 的 getWorkers()：它靠 Redis 連線名稱辨識 worker，Node 版用 base64 的佇列名稱，Python 版不是，對不上。
 */
export const REC_WORKER_ONLINE_KEY = "rec-worker:online";
/**
 * 只送一次：worker 取到 job 就把照片清掉，重試也沒有照片可用；漏掉的由補排重送。
 * 成功或失敗都立刻刪除，照片不留在 Redis。jobId 用照片 id，同一張照片還在排隊時不會重複排。
 */
const JOB_OPTIONS = { attempts: 1, removeOnComplete: true, removeOnFail: true };
/**
 * 補排還沒處理過的主照片（worker 停機期間上傳的、worker 上線前就有的）：每分鐘一批，
 * 佇列清空才排下一批，Redis 裡最多同時放一批照片。
 */
const BACKFILL_INTERVAL_MS = 60_000;
const BACKFILL_BATCH = 20;

const resultSchema = z.object({
  photoId: z.string().uuid(),
  modelVersion: z.string().min(1).max(200),
  // 找不到臉時是 null。
  embedding: z
    .array(z.number().finite())
    .length(APPEARANCE_DIMENSIONS)
    .nullable(),
});

@Injectable()
export class RecJobs implements OnModuleInit, OnModuleDestroy {
  /** BullMQ 要求 maxRetriesPerRequest 是 null，不能共用 Infrastructure 的 Redis 連線。 */
  private readonly connection = {
    url: config.REDIS_URL,
    maxRetriesPerRequest: null,
  };
  readonly queue = new Queue(REC_JOBS_QUEUE, { connection: this.connection });
  private results?: Worker;
  private backfillTimer?: NodeJS.Timeout;

  constructor(
    private db: Database,
    private infra: Infrastructure,
    private vectors: VectorStore,
  ) {}

  onModuleInit() {
    // 沒接 error 事件時 Redis 一斷線整個程序就會結束；推薦是附加功能，只記錄固定字串。
    this.queue.on("error", () => console.error("rec_jobs_queue_error"));
    this.results = new Worker(
      REC_RESULTS_QUEUE,
      (job) => this.handleResult(job),
      { connection: this.connection, concurrency: 2 },
    );
    this.results.on("error", () => console.error("rec_results_worker_error"));
    this.results.on("failed", (_job, error) =>
      console.error("rec_result_failed", error?.message ?? "unknown"),
    );
    this.backfillTimer = setInterval(
      () =>
        void this.backfill().catch(() => console.error("rec_backfill_failed")),
      BACKFILL_INTERVAL_MS,
    );
  }

  async onModuleDestroy() {
    clearInterval(this.backfillTimer);
    await this.results?.close();
    await this.queue.close();
  }

  /**
   * 排一張照片去算外貌向量。image 是剛上傳、存進物件儲存的同一份 JPEG；沒給就從物件儲存讀。
   * worker 沒在跑就不排，回傳 false。呼叫端不等待：算向量失敗不能影響上傳或刪照片。
   */
  async enqueueAppearance(
    photo: { id: string; storageKey: string },
    image?: Buffer,
  ) {
    if (!(await this.workerOnline())) return false;
    await this.add(photo, image);
    return true;
  }

  /** 補排一批還沒處理過的主照片（新的在前）；worker 沒在跑、或上一批還沒做完就不排。回傳排了幾張。 */
  private async backfill() {
    if (!(await this.workerOnline())) return 0;
    if (
      await this.queue.getJobCountByTypes(
        "waiting",
        "active",
        "delayed",
        "prioritized",
      )
    )
      return 0;
    const photos = await this.db.photo.findMany({
      where: { isAvatar: true, deletedAt: null, appearanceEmbedding: null },
      orderBy: { createdAt: "desc" },
      take: BACKFILL_BATCH,
      select: { id: true, storageKey: true },
    });
    let queued = 0;
    for (const photo of photos)
      try {
        await this.add(photo);
        queued++;
      } catch {
        console.error("rec_backfill_photo_failed");
      }
    return queued;
  }

  private async workerOnline() {
    return (await this.infra.redis.exists(REC_WORKER_ONLINE_KEY)) > 0;
  }

  private async add(photo: { id: string; storageKey: string }, image?: Buffer) {
    const source = image ?? (await this.read(photo.storageKey));
    await this.queue.add(
      EMBED_APPEARANCE,
      { photoId: photo.id, image: source.toString("base64") },
      { ...JOB_OPTIONS, jobId: `appearance-${photo.id}` },
    );
  }

  private async read(key: string) {
    const stream = await this.infra.storage.getObject(config.S3_BUCKET, key);
    const chunks: Buffer[] = [];
    for await (const chunk of stream) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks);
  }

  /**
   * worker 的結果：有向量時扣掉同版本的戴眼鏡方向再存；找不到臉也存一列（embedding 是 null，
   * 這個人的外貌分數給中間值），定期補排就不會一直重送同一張照片。
   * 照片在算的期間被刪掉就不寫（storeAppearanceEmbedding 會檢查）。
   * 錯誤只丟固定代碼：驗證錯誤的訊息會帶到向量內容。
   */
  async handleResult(job: Job) {
    if (job.name !== EMBED_APPEARANCE)
      throw new Error("UNSUPPORTED_RESULT_JOB");
    const parsed = resultSchema.safeParse(job.data);
    if (!parsed.success) throw new Error("INVALID_REC_RESULT");
    const { photoId, modelVersion, embedding } = parsed.data;
    const glasses = embedding
      ? await this.vectors.appearanceDirection("glasses")
      : null;
    if (embedding && glasses?.modelVersion === modelVersion)
      await this.vectors.storeAppearanceEmbedding(
        photoId,
        `${modelVersion}+glasses-project`,
        projectOut(embedding, glasses.vector),
      );
    else
      await this.vectors.storeAppearanceEmbedding(
        photoId,
        modelVersion,
        embedding,
      );
  }
}
