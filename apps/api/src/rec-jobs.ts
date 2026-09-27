// 外貌向量的背景工作：新的主照片 → rec-jobs（services/recommendation 的 worker 算向量）
// → rec-results → 這裡寫進 appearance_embeddings（ADR 0002：只有 NestJS 碰資料庫）。
//
// 佇列和 AI 服務的 ai-jobs／ai-results 分開：兩邊的 worker 遇到不認得的 job 都會直接失敗。
// 照片原檔放進 job（上傳時已轉成長邊 1600px 以內的 JPEG），不再縮圖或重新壓縮：ml/ 離線算種子向量時
// 讀的也是原檔，兩邊的向量才一致（重新壓縮會讓少數照片的去背範圍改變）。worker 取到 job 就把照片從 Redis 清掉。
// worker 沒在跑時不排（照片不會堆在 Redis），等它上線後由每分鐘的補排接手。
//
// 測試畫面的「像在哪裡」（explain-appearance）也走同一組佇列：點開並排比較時才排，結果只放 Redis 1 小時。
import { Injectable, OnModuleDestroy, OnModuleInit } from "@nestjs/common";
import { Queue, Worker, type Job } from "bullmq";
import { z } from "zod";
import { Database, Infrastructure, config } from "./core";
import { APPEARANCE_DIMENSIONS, VectorStore } from "./ai-store";
import { projectOut } from "./discovery-rank";

export const REC_JOBS_QUEUE = "rec-jobs";
export const REC_RESULTS_QUEUE = "rec-results";
export const EMBED_APPEARANCE = "embed-appearance";
export const EXPLAIN_APPEARANCE = "explain-appearance";
/** 臉部區域；和 services/recommendation/appearance/explain.py 的 REGIONS 一致。左右照畫面來分。 */
export const FACE_REGIONS = [
  "eyebrows",
  "eyes",
  "nose",
  "lips",
  "left_cheek",
  "right_cheek",
  "chin",
] as const;
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
/**
 * 「像在哪裡」的結果只放 Redis、不寫資料庫。排進佇列時先放「計算中」標記：前端每秒輪詢不會重複排，
 * worker 當掉時標記過期後可以重排。鍵用兩張主照片的 id，換了主照片自然對不到舊結果。
 */
const EXPLAIN_TTL_SECONDS = 60 * 60;
const EXPLAIN_PENDING_SECONDS = 60;
const EXPLAIN_PENDING = "pending";
const explainKey = (candidatePhotoId: string, anchorPhotoId: string) =>
  `rec:explain:${candidatePhotoId}:${anchorPhotoId}`;
const UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

const resultSchema = z.object({
  photoId: z.string().uuid(),
  modelVersion: z.string().min(1).max(200),
  // 找不到臉時是 null。
  embedding: z
    .array(z.number().finite())
    .length(APPEARANCE_DIMENSIONS)
    .nullable(),
});
// [x, y, w, h]，原始照片的像素。
const box = z.tuple([
  z.number().int().min(0),
  z.number().int().min(0),
  z.number().int().positive(),
  z.number().int().positive(),
]);
const imageSize = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive(),
});
const explanationSchema = z.object({
  similarity: z.number().finite(),
  candidate: imageSize,
  anchor: imageSize,
  // 依下降量由大到小；drop 是兩張臉各遮一次的平均。
  regions: z
    .array(
      z.object({
        name: z.enum(FACE_REGIONS),
        drop: z.number().finite(),
        dropCandidate: z.number().finite(),
        dropAnchor: z.number().finite(),
        candidate: z.array(box).max(2),
        anchor: z.array(box).max(2),
      }),
    )
    .max(FACE_REGIONS.length),
});
export type AppearanceExplanation = z.infer<typeof explanationSchema>;
const explainResultSchema = z.object({
  requestId: z.string().regex(new RegExp(`^${UUID}:${UUID}$`)),
  // 任一張找不到臉或臉部特徵點時是 null。
  result: explanationSchema.nullable(),
});
type Photo = { id: string; storageKey: string };

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
  async enqueueAppearance(photo: Photo, image?: Buffer) {
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

  /**
   * 測試畫面「像在哪裡」：有結果就回傳；沒有就排給 worker、回傳 pending，前端每秒再問一次。
   * worker 沒在跑回傳 offline。照片和算向量一樣送原檔，戴眼鏡方向也一起送，相似度才會和排序用的一致。
   */
  async explainAppearance(
    candidate: Photo,
    anchor: Photo,
  ): Promise<
    | { status: "ready"; result: AppearanceExplanation | null }
    | { status: "pending" }
    | { status: "offline" }
  > {
    const key = explainKey(candidate.id, anchor.id);
    const cached = await this.infra.redis.get(key);
    if (cached === EXPLAIN_PENDING) return { status: "pending" };
    if (cached) return { status: "ready", result: JSON.parse(cached) };
    if (!(await this.workerOnline())) return { status: "offline" };
    const claimed = await this.infra.redis.set(
      key,
      EXPLAIN_PENDING,
      "EX",
      EXPLAIN_PENDING_SECONDS,
      "NX",
    );
    if (!claimed) return { status: "pending" };
    try {
      const [candidateImage, anchorImage, glasses] = await Promise.all([
        this.read(candidate.storageKey),
        this.read(anchor.storageKey),
        this.vectors.appearanceDirection("glasses"),
      ]);
      await this.queue.add(
        EXPLAIN_APPEARANCE,
        {
          requestId: `${candidate.id}:${anchor.id}`,
          candidate: candidateImage.toString("base64"),
          anchor: anchorImage.toString("base64"),
          ...(glasses && {
            direction: glasses.vector,
            directionVersion: glasses.modelVersion,
          }),
        },
        { ...JOB_OPTIONS, jobId: `explain-${candidate.id}-${anchor.id}` },
      );
    } catch (error) {
      await this.infra.redis.del(key);
      throw error;
    }
    return { status: "pending" };
  }

  private async workerOnline() {
    return (await this.infra.redis.exists(REC_WORKER_ONLINE_KEY)) > 0;
  }

  private async add(photo: Photo, image?: Buffer) {
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
    if (job.name === EXPLAIN_APPEARANCE) return this.storeExplanation(job.data);
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

  /** 「像在哪裡」的結果放進 Redis 1 小時，取代計算中標記。 */
  private async storeExplanation(data: unknown) {
    const parsed = explainResultSchema.safeParse(data);
    if (!parsed.success) throw new Error("INVALID_REC_RESULT");
    const [candidatePhotoId, anchorPhotoId] = parsed.data.requestId.split(":");
    await this.infra.redis.set(
      explainKey(candidatePhotoId, anchorPhotoId),
      JSON.stringify(parsed.data.result),
      "EX",
      EXPLAIN_TTL_SECONDS,
    );
  }
}
