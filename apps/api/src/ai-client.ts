// 呼叫 Python AI 服務（services/ai）的唯一入口。
//
// 契約見 docs/ai/REPLY-SUGGESTIONS-SPEC.md 第 7 節：所有路徑都在 /internal/ai/ 底下、
// 需要 X-Internal-Token、請求與回應都是 camelCase JSON。
// 可預期的錯誤回 503 與固定代碼（LLM_*／EMBEDDING_*／EXTRACTION_*），
// 這裡把它們轉成 AiError，讓呼叫端自己決定要「整個失敗」還是「降級繼續」。
import { Injectable } from "@nestjs/common";
import { z } from "zod";
import { config } from "./core";

/** 產生推薦整次請求的逾時；AI 服務內部的總逾時是 25 秒，這裡多留 5 秒給網路與排隊。 */
const SUGGEST_TIMEOUT_MS = 30_000;
/** 向量化的逾時；AI 服務每批的逾時是 25 秒。 */
const EMBED_TIMEOUT_MS = 30_000;

/**
 * AI 服務可預期的錯誤。
 * code 是 AI 服務回的固定代碼（例如 LLM_UNAVAILABLE），或這裡自己判斷出來的狀況：
 * - AI_UNREACHABLE：連不上（服務沒開、網路不通、逾時）。
 * - AI_BAD_RESPONSE：回應格式不符合契約。
 * - AI_REJECTED：token 錯誤或請求被判為格式錯誤（401／422），屬於程式問題。
 */
export class AiError extends Error {
  constructor(
    readonly code: string,
    readonly status = 503,
  ) {
    super(code);
    this.name = "AiError";
  }
}

/** 訊息來源；與資料庫的 message_origins.origin 及 AI 服務的 Origin 相同。 */
export type AiOrigin = "human" | "ai_verbatim" | "ai_edited";
/** 推薦的用途；也用來判斷 AI 訊息有沒有「開啟新話題」。 */
export type AiIntent =
  | "answer"
  | "question"
  | "callback"
  | "humor"
  | "plan"
  | "share";

/** 使用者檔案快照：標籤傳中文顯示名稱（例如「登山」），不是資料庫代碼。 */
export type AiProfile = {
  displayName: string;
  age?: number | null;
  gender?: string | null;
  city?: string | null;
  bio?: string;
  occupation?: string | null;
  education?: string | null;
  heightCm?: number | null;
  interests?: string[];
  hobbies?: string[];
  foods?: string[];
  traits?: string[];
  datingGoals?: string[];
};

/** 產生推薦時用的一則訊息；sender 只分 A（按按鈕的人）與 B（聊天對象）。 */
export type AiChatMessage = {
  id: string;
  sender: "A" | "B";
  content: string;
  createdAt: string;
  origin: AiOrigin;
};

/** 背景工作用的一則訊息；站在整個聊天室的角度，所以用 id 與暱稱標示說話者。 */
export type AiIndexMessage = {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  createdAt: string;
  origin: AiOrigin;
  suggestionIntent?: AiIntent | null;
};

/** 萃取風格卡時用的一則訊息（某個人自己發出的）。 */
export type AiOwnMessage = {
  id: string;
  conversationId: string;
  content: string;
  createdAt: string;
  origin: AiOrigin;
  inAiTopic: boolean;
};

/** 寫法的數值統計；欄位與 AI 服務的 StyleStats 相同。 */
export type AiStyleStats = {
  messageCount?: number;
  medianChars: number;
  meanChars: number;
  emojiPerMessage: number;
  questionRatio: number;
  exclamationRatio: number;
  laughterRatio: number;
  burstMean?: number;
  particles?: Record<string, number>;
};

/** 風格卡上的一條抽象特徵句。 */
export type AiStyleFacet = {
  kind: "topic" | "tone" | "habit" | "avoid";
  statement: string;
  weight: number;
  evidence: number;
};

/** 一位使用者的風格卡；由 build-style 背景工作產生後存進 user_style_profiles。 */
export type AiStyleCard = {
  featureVersion: string;
  userId?: string | null;
  confidence: "high" | "low" | "none";
  sampleSource: "chat" | "mixed" | "bio" | "none";
  stats: AiStyleStats;
  voiceNotes: string[];
  facets: AiStyleFacet[];
  bioSample: string;
  messageCount: number;
  windowFrom?: string | null;
  windowTo?: string | null;
  modelName?: string | null;
  promptVersion?: string | null;
};

/** POST /internal/ai/reply-suggestions 的請求（由 NestJS 把資料庫查到的東西組好）。 */
export type AiSuggestRequest = {
  requestId: string;
  requester: AiProfile;
  partner: AiProfile;
  sharedTags: string[];
  recentMessages: AiChatMessage[];
  retrievedChunks: { content: string; lastAt?: string; score?: number }[];
  conversationSummary?: string | null;
  requesterStyle?: AiStyleCard | null;
  partnerStyle?: AiStyleCard | null;
  partnerFacets: AiStyleFacet[];
  siteStats?: AiStyleStats | null;
  excludeTexts: string[];
};

/** POST /internal/ai/chunks 的請求（也是 chunk-embed 這個 job 的資料）。 */
export type AiChunkRequest = {
  conversationId: string;
  messages: AiIndexMessage[];
  embed: boolean;
};

/** POST /internal/ai/topic-spans 的請求（也是 topic-spans 這個 job 的資料）。 */
export type AiTopicSpanRequest = {
  conversationId: string;
  messages: AiIndexMessage[];
};

/** POST /internal/ai/conversation-summary 的請求（也是 summarize 這個 job 的資料）。 */
export type AiSummaryRequest = {
  conversationId: string;
  previousSummary?: string | null;
  messages: AiIndexMessage[];
  maxChars: number;
};

/** POST /internal/ai/style-profile 的請求（也是 build-style 這個 job 的資料）。 */
export type AiStyleProfileRequest = {
  userId: string;
  bio: string;
  messages: AiOwnMessage[];
  windowFrom?: string | null;
  windowTo?: string | null;
  capped: boolean;
  siteStats?: AiStyleStats | null;
  embedFacets: boolean;
};

// ---------------------------------------------------------------------------
// 回應格式：只驗證後端真的會用到的欄位，AI 服務多回欄位不會讓這裡壞掉。
// ---------------------------------------------------------------------------

const styleStats = z.object({
  messageCount: z.number().optional(),
  medianChars: z.number(),
  meanChars: z.number(),
  emojiPerMessage: z.number(),
  questionRatio: z.number(),
  exclamationRatio: z.number(),
  laughterRatio: z.number(),
  burstMean: z.number().optional(),
  particles: z.record(z.string(), z.number()).optional(),
});
const styleFacet = z.object({
  kind: z.enum(["topic", "tone", "habit", "avoid"]),
  statement: z.string(),
  weight: z.number(),
  evidence: z.number().int(),
});
const styleCard = z.object({
  featureVersion: z.string(),
  userId: z.string().nullish(),
  confidence: z.enum(["high", "low", "none"]),
  sampleSource: z.enum(["chat", "mixed", "bio", "none"]),
  stats: styleStats,
  voiceNotes: z.array(z.string()),
  facets: z.array(styleFacet),
  bioSample: z.string(),
  messageCount: z.number().int(),
  windowFrom: z.string().nullish(),
  windowTo: z.string().nullish(),
  modelName: z.string().nullish(),
  promptVersion: z.string().nullish(),
});
const embedResponse = z.object({
  model: z.string(),
  dimensions: z.number().int(),
  vectors: z.array(z.array(z.number())),
});
const suggestResponse = z.object({
  requestId: z.string(),
  status: z.enum(["ok", "partial", "empty"]),
  mode: z.enum(["opener", "reply", "follow_up", "revive"]),
  suggestions: z.array(
    z.object({
      rank: z.number().int(),
      text: z.string(),
      intent: z.enum([
        "answer",
        "question",
        "callback",
        "humor",
        "plan",
        "share",
      ]),
      styleTarget: z.enum(["partner", "blend"]),
      styleDistance: z.number(),
      reason: z.string().default(""),
    }),
  ),
  rejected: z.array(z.object({ text: z.string(), reasonCode: z.string() })),
  notice: z.string().nullish(),
  modelName: z.string().nullish(),
  promptVersion: z.string(),
  usage: z.object({
    inputTokens: z.number().int(),
    outputTokens: z.number().int(),
    requests: z.number().int(),
  }),
});
/** chunk-embed 的結果（由 AI worker 放進 ai-results，格式與 POST /chunks 的回應相同）。 */
export const chunkResult = z.object({
  conversationId: z.string(),
  chunkVersion: z.string(),
  embeddingModel: z.string().nullish(),
  chunks: z.array(
    z.object({
      firstMessageId: z.string(),
      lastMessageId: z.string(),
      firstAt: z.string(),
      lastAt: z.string(),
      messageCount: z.number().int(),
      content: z.string(),
      tokenEstimate: z.number().int(),
      isOpen: z.boolean(),
      vector: z.array(z.number()).nullish(),
    }),
  ),
});
/** topic-spans 的結果。 */
export const topicSpanResult = z.object({
  conversationId: z.string(),
  detectorVersion: z.string(),
  usedVectors: z.boolean(),
  spans: z.array(
    z.object({
      initiatingMessageId: z.string(),
      initiatorId: z.string(),
      startMessageId: z.string(),
      endMessageId: z.string(),
      messageCount: z.number().int(),
      endReason: z.enum(["time_gap", "topic_shift", "cap", "end_of_data"]),
    }),
  ),
});
/** summarize 的結果。 */
export const summaryResult = z.object({
  conversationId: z.string(),
  summary: z.string(),
  untilMessageId: z.string(),
  modelName: z.string().nullish(),
  promptVersion: z.string(),
});
/** build-style 的結果；facetVectors 與 card.facets 一一對應。 */
export const styleProfileResult = z.object({
  card: styleCard,
  facetVectors: z.array(z.array(z.number())).nullish(),
  embeddingModel: z.string().nullish(),
  rejectedStatements: z.number().int().default(0),
});

export type AiSuggestResponse = z.infer<typeof suggestResponse>;
export type AiChunkResult = z.infer<typeof chunkResult>;
export type AiTopicSpanResult = z.infer<typeof topicSpanResult>;
export type AiSummaryResult = z.infer<typeof summaryResult>;
export type AiStyleProfileResult = z.infer<typeof styleProfileResult>;

@Injectable()
export class AiClient {
  /**
   * 送一個請求給 AI 服務並驗證回應。
   *
   * 錯誤一律轉成 AiError：
   * - 連不上或逾時 → AI_UNREACHABLE（不會把原始例外往外丟，避免日誌出現內部位址）。
   * - 503 → 用回應裡的固定代碼（LLM_UNAVAILABLE…），呼叫端可據此顯示「AI 忙碌中」。
   * - 401／422 → AI_REJECTED，代表 token 或請求格式有問題，屬於程式或設定錯誤。
   * - 其他狀態或格式不符 → AI_BAD_RESPONSE。
   */
  private async post<T>(
    path: string,
    body: unknown,
    schema: z.ZodType<T>,
    timeoutMs: number,
  ): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${config.AI_INTERNAL_URL}/internal/ai/${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Internal-Token": config.AI_INTERNAL_TOKEN,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch {
      throw new AiError("AI_UNREACHABLE");
    }
    if (!response.ok) {
      // 503 的內容是 {code, message}；其他狀態不一定有 JSON，解析失敗就用通用代碼。
      const payload = await response.json().catch(() => null);
      const code =
        payload && typeof payload === "object" && "code" in payload
          ? String((payload as { code: unknown }).code)
          : null;
      if (response.status === 503 && code) throw new AiError(code);
      if (response.status === 401 || response.status === 422)
        throw new AiError("AI_REJECTED");
      throw new AiError("AI_BAD_RESPONSE");
    }
    const parsed = schema.safeParse(await response.json().catch(() => null));
    if (!parsed.success) throw new AiError("AI_BAD_RESPONSE");
    return parsed.data;
  }

  /**
   * 把文字轉成向量。
   * purpose=query 用在檢索時的查詢句，document 用在要存進資料庫的內容；
   * 兩者的前綴不同，混用會讓檢索變差（見 AI 服務的 format_for_embedding）。
   */
  async embed(texts: string[], purpose: "query" | "document", title?: string) {
    return this.post(
      "embed",
      { texts, purpose, ...(title ? { title } : {}) },
      embedResponse,
      EMBED_TIMEOUT_MS,
    );
  }

  /** 產生 3～5 則推薦；請求內容由 ai-reply.ts 組好（它負責權限檢查與資料庫查詢）。 */
  async suggest(request: AiSuggestRequest) {
    return this.post(
      "reply-suggestions",
      request,
      suggestResponse,
      SUGGEST_TIMEOUT_MS,
    );
  }
}
