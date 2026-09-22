// 把資料庫裡的資料組成 AI 服務要的請求格式。
//
// AI 服務不連資料庫（ADR 0002），所以「要給模型看什麼」全部在這裡決定：
// 檔案快照、聊天室近期訊息、背景工作要處理的訊息、萃取風格卡的訊息視窗、全站平均。
import { Injectable } from "@nestjs/common";
import { Database, Infrastructure, ageAt } from "./core";
import { VectorStore } from "./ai-store";
import type {
  AiChatMessage,
  AiIndexMessage,
  AiIntent,
  AiOrigin,
  AiOwnMessage,
  AiProfile,
  AiStyleStats,
} from "./ai-client";

/** 產生推薦時放進 prompt 的近期訊息則數；AI 服務會再依 token 預算截斷。 */
export const RECENT_MESSAGES = 60;
/** 背景工作一次最多處理的訊息則數（AI 服務的上限是 20,000）。 */
export const JOB_MESSAGES = 2000;
/** 風格卡視窗：訊息數在這個數字以內就全部使用（規格 5.3）。 */
export const WINDOW_BASE = 5000;
/** 風格卡視窗：超過 WINDOW_BASE 時，「最近 5,000 則」與「最近 90 天」取較多者。 */
export const WINDOW_DAYS = 90;
/** 風格卡視窗的硬上限，避免一次工作塞進太多訊息。 */
export const WINDOW_CAP = 20000;
/** 全站平均的快取秒數：這個數字變動很慢，每天算一次就夠。 */
const SITE_STATS_TTL = 24 * 60 * 60;
/** 全站平均的快取鍵。 */
const SITE_STATS_KEY = "ai:site-stats";

/** 只有這裡在做「使用者 id → A／B」的轉換，AI 服務完全不知道使用者 id。 */
const roleOf = (senderId: string, requesterId: string): "A" | "B" =>
  senderId === requesterId ? "A" : "B";

/** 沒有 message_origins 紀錄的訊息一律當作真人打的（這張表是後來才加的）。 */
const originOf = (origin: { origin: string } | null): AiOrigin =>
  origin?.origin === "ai_verbatim" || origin?.origin === "ai_edited"
    ? origin.origin
    : "human";

@Injectable()
export class AiData {
  constructor(
    private db: Database,
    private infra: Infrastructure,
    private vectors: VectorStore,
  ) {}

  /**
   * 把使用者資料變成檔案快照。
   *
   * 標籤一律轉成中文顯示名稱（例如 hiking → 登山），因為模型看的是中文；
   * 傳代碼會讓模型看不懂，也會讓 prompt 出現英文詞彙。
   */
  profile(user: {
    profile: {
      displayName: string;
      birthDate: Date;
      gender: string;
      city: string;
      bio: string;
      occupation: string | null;
      education: string | null;
      heightCm: number | null;
      interests: string[];
      hobbies: string[];
      foods: string[];
    } | null;
    traits: { trait: { category: string; code: string; labelZh: string | null } }[];
  }): AiProfile {
    const p = user.profile;
    const labels = (datingGoal: boolean) =>
      user.traits
        .filter((t) => (t.trait.category === "dating_goal") === datingGoal)
        .map((t) => t.trait.labelZh || t.trait.code);
    return {
      displayName: p?.displayName || "使用者",
      age: p ? ageAt(p.birthDate) : null,
      gender: p?.gender ?? null,
      city: p?.city ?? null,
      bio: p?.bio ?? "",
      occupation: p?.occupation ?? null,
      education: p?.education ?? null,
      heightCm: p?.heightCm ?? null,
      interests: p?.interests ?? [],
      hobbies: p?.hobbies ?? [],
      foods: p?.foods ?? [],
      traits: labels(false),
      datingGoals: labels(true),
    };
  }

  /**
   * 兩個人共同的標籤，prompt 會用它找話題。
   * 只比中文顯示名稱，所以 traits（登山）與 interests（登山）重複時會自動合併。
   */
  sharedTags(a: AiProfile, b: AiProfile) {
    const setOf = (profile: AiProfile) =>
      new Set([
        ...(profile.interests ?? []),
        ...(profile.hobbies ?? []),
        ...(profile.foods ?? []),
        ...(profile.traits ?? []),
      ]);
    const other = setOf(b);
    return [...setOf(a)].filter((tag) => other.has(tag)).slice(0, 60);
  }

  /**
   * 這個聊天室最近的訊息（舊到新），附上來源標記。
   *
   * AI 產生的訊息在上下文一律保留（規格 5.1）：它們真的送出過、對方也看到了，
   * 只是模型會知道那幾則是 AI 推薦，避免把自己的話誤認成使用者的寫法。
   */
  async chatMessages(
    conversationId: string,
    requesterId: string,
    limit = RECENT_MESSAGES,
  ): Promise<AiChatMessage[]> {
    const rows = await this.db.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: "desc" },
      take: limit,
      include: { origin: true },
    });
    return rows.reverse().map((row) => ({
      id: row.id,
      sender: roleOf(row.senderId, requesterId),
      content: row.content,
      createdAt: row.createdAt.toISOString(),
      origin: originOf(row.origin),
    }));
  }

  /**
   * 背景工作要處理的訊息（切片、話題區段、摘要共用）。
   *
   * 與 chatMessages 的差別：背景工作處理的是整個聊天室，不站在某個人的角度，
   * 所以帶的是使用者 id 與暱稱。suggestionIntent 只有 AI 訊息才有，
   * 話題區段偵測要用它判斷那則推薦是不是「開啟了新話題」（規格 5.3）。
   *
   * `after` 是上次處理到的位置：切片從「還沒結束的片段」開始、摘要從上次涵蓋到的訊息之後開始。
   */
  async indexMessages(
    conversationId: string,
    after: { at: Date; inclusive: boolean } | null,
    limit = JOB_MESSAGES,
  ): Promise<AiIndexMessage[]> {
    const rows = await this.db.message.findMany({
      where: {
        conversationId,
        ...(after
          ? { createdAt: after.inclusive ? { gte: after.at } : { gt: after.at } }
          : {}),
      },
      // 先取最近的 limit 則再反轉：聊天室很長時只處理尾端，不會把整段歷史塞進 job。
      orderBy: { createdAt: "desc" },
      take: limit,
      include: {
        origin: { include: { suggestion: true } },
        sender: { include: { profile: true } },
      },
    });
    return rows.reverse().map((row) => ({
      id: row.id,
      senderId: row.senderId,
      senderName: row.sender.profile?.displayName || "使用者",
      content: row.content,
      createdAt: row.createdAt.toISOString(),
      origin: originOf(row.origin),
      suggestionIntent: (row.origin?.suggestion?.intent as AiIntent) ?? null,
    }));
  }

  /**
   * 萃取風格卡要用的訊息（規格 5.3 的視窗規則）。
   *
   * 順序很重要：**先排除 AI 來源的訊息，再套視窗**，否則 AI 訊息會佔掉視窗名額。
   * - 真人訊息 ≤ 5,000 則：全部使用。
   * - 超過：「最近 5,000 則」與「最近 90 天」取「則數較多」的那一種。
   * - 不論哪種，最多 20,000 則（capped 會記進風格卡，方便日後判讀）。
   *
   * inAiTopic 由 ai_topic_spans 決定：只有「這個人自己發起的 AI 話題區段」裡、
   * 他本人發的訊息才會標記，寫法照用、話題喜好降權。
   */
  async ownMessages(userId: string) {
    const humanOnly = {
      senderId: userId,
      OR: [{ origin: { is: null } }, { origin: { origin: "human" } }],
    };
    const total = await this.db.message.count({ where: humanOnly });
    const cutoff = new Date(Date.now() - WINDOW_DAYS * 24 * 60 * 60 * 1000);
    let take = total;
    let from: Date | undefined;
    if (total > WINDOW_BASE) {
      const recentDays = await this.db.message.count({
        where: { ...humanOnly, createdAt: { gte: cutoff } },
      });
      // 取「則數較多」的那一邊：90 天內的訊息比 5,000 則多就用時間視窗，否則用最近 5,000 則。
      if (recentDays > WINDOW_BASE) {
        take = Math.min(recentDays, WINDOW_CAP);
        from = cutoff;
      } else {
        take = WINDOW_BASE;
      }
    }
    take = Math.min(take, WINDOW_CAP);
    const rows = await this.db.message.findMany({
      where: { ...humanOnly, ...(from ? { createdAt: { gte: from } } : {}) },
      orderBy: { createdAt: "desc" },
      take,
      select: {
        id: true,
        conversationId: true,
        content: true,
        createdAt: true,
      },
    });
    const ordered = rows.reverse();
    const aiTopics = await this.vectors.aiTopicMessageIds(
      userId,
      ordered.length ? ordered[0].createdAt : null,
    );
    const messages: AiOwnMessage[] = ordered.map((row) => ({
      id: row.id,
      conversationId: row.conversationId,
      content: row.content,
      createdAt: row.createdAt.toISOString(),
      // 這裡的訊息已經只剩真人訊息，origin 固定是 human；AI 服務會再檢查一次當保險。
      origin: "human",
      inAiTopic: aiTopics.has(row.id),
    }));
    return {
      messages,
      windowFrom: ordered.length ? ordered[0].createdAt.toISOString() : null,
      windowTo: ordered.length
        ? ordered[ordered.length - 1].createdAt.toISOString()
        : null,
      // capped：視窗規則算出來的則數被 20,000 的硬上限砍掉了。
      capped: total > take,
      humanCount: total,
    };
  }

  /**
   * 全站的聊天寫法平均值，冷啟動時當數值目標（規格 5.3、5.11）。
   *
   * 結果快取一天：這個數字變動很慢，但每次都算會掃 5,000 則訊息。
   * Redis 讀寫失敗時直接重算，不讓快取問題影響推薦。
   * emoji 與連發則數在 SQL 裡不好算，沿用 AI 服務的內建值（同一個資料庫量出來的）。
   */
  async siteStats(): Promise<AiStyleStats | null> {
    const cached = await this.infra.redis.get(SITE_STATS_KEY).catch(() => null);
    if (cached) {
      try {
        return JSON.parse(cached) as AiStyleStats;
      } catch {
        // 快取內容壞掉（例如格式改版）就當作沒有快取，重新計算。
      }
    }
    const row = await this.vectors.siteStats();
    if (!row) return null;
    const stats: AiStyleStats = {
      messageCount: row.messageCount,
      medianChars: row.medianChars,
      meanChars: row.meanChars,
      emojiPerMessage: 0.15,
      questionRatio: row.questionRatio,
      exclamationRatio: row.exclamationRatio,
      laughterRatio: row.laughterRatio,
      burstMean: 1.5,
      particles: {},
    };
    await this.infra.redis
      .set(SITE_STATS_KEY, JSON.stringify(stats), "EX", SITE_STATS_TTL)
      .catch(() => null);
    return stats;
  }
}
