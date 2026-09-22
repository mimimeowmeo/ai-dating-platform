// 「AI 推薦回覆」的同步流程（規格 6.1）。
//
// 使用者按一次按鈕，後端做這幾件事：
//   1. access() 權限檢查，讀近期訊息、雙方檔案、風格卡、摘要。
//   2. 把最近幾則訊息送去 /embed 變成查詢向量。
//   3. 用查詢向量在 pgvector 找出相關的舊對話片段與對方的特徵句。
//   4. 呼叫 /reply-suggestions 產生 3～5 則推薦。
//   5. 把請求與「全部」推薦（含被刪掉的候選）寫進資料庫，再回傳前端。
//
// 向量化失敗時不讓整件事失敗：少了 RAG 的推薦仍然可用，只是少了舊話題的線索。
import { Injectable } from "@nestjs/common";
import { randomUUID } from "node:crypto";
import { Database, Infrastructure, fail } from "./core";
import { userInclude } from "./profiles";
import { Social } from "./social";
import { AiClient, AiError } from "./ai-client";
import type { AiStyleFacet, AiSuggestRequest } from "./ai-client";
import { AiData } from "./ai-data";
import { AiJobs } from "./ai-jobs";
import { VectorStore } from "./ai-store";

/**
 * 每分鐘最多按幾次。
 * 這不是產品配額（決策：每人不限次數），只是避免按住按鈕或腳本連打把 AI 額度用光；
 * Ollama Cloud 免費方案一次只處理 1 個請求，連打也只會讓自己排隊。
 */
const SUGGEST_PER_MINUTE = 30;
/** 組查詢向量時取最近幾則訊息：太多會讓查詢向量失焦，抓不到目前的話題。 */
const QUERY_MESSAGES = 6;
/** 「換一批」時要避開的舊推薦則數上限（AI 服務的上限是 30）。 */
const EXCLUDE_LIMIT = 30;

@Injectable()
export class ReplySuggestions {
  constructor(
    private db: Database,
    private infra: Infrastructure,
    private social: Social,
    private data: AiData,
    private vectors: VectorStore,
    private ai: AiClient,
    private jobs: AiJobs,
  ) {}

  /**
   * 產生 3～5 則推薦並全部存檔，回傳給前端。
   *
   * 回傳的每一則都帶 id：使用者選了之後，送訊息時把它當 suggestionId 一起送，
   * 後端才能標記訊息來源（ai-origins.ts）。
   */
  async suggest(userId: string, conversationId: string) {
    await this.infra.limit(`ai-suggest:${userId}`, SUGGEST_PER_MINUTE, 60);
    const conversation = await this.social.access(userId, conversationId);
    const started = Date.now();
    const requestId = randomUUID();

    // --- 1. 讀出要給模型看的資料 ---------------------------------------
    const users = await this.db.user.findMany({
      where: { id: { in: [userId, conversation.otherUserId] } },
      include: userInclude,
    });
    const requesterRow = users.find((u) => u.id === userId);
    const partnerRow = users.find((u) => u.id === conversation.otherUserId);
    if (!requesterRow || !partnerRow)
      return fail(404, "NOT_FOUND", "找不到聊天對象。");
    const requester = this.data.profile(requesterRow);
    const partner = this.data.profile(partnerRow);
    const [recentMessages, requesterStyle, partnerStyle, summary] =
      await Promise.all([
        this.data.chatMessages(conversationId, userId),
        this.vectors.styleCard(userId),
        this.vectors.styleCard(conversation.otherUserId),
        this.db.conversationSummary.findFirst({
          where: { conversationId },
          orderBy: { createdAt: "desc" },
          select: { summary: true },
        }),
      ]);
    const lastMessage = recentMessages.at(-1) ?? null;

    // --- 2～3. 查詢向量 → pgvector 檢索 --------------------------------
    // 沒有訊息（剛配對）時改用對方的自我介紹當查詢，至少能找到相關的特徵句。
    const queryText =
      recentMessages
        .slice(-QUERY_MESSAGES)
        .map((message) => message.content)
        .join("\n") || partner.bio ||
      [...(partner.interests ?? []), ...(partner.traits ?? [])].join("、");
    let chunks: AiSuggestRequest["retrievedChunks"] = [];
    let partnerFacets: AiStyleFacet[] = [];
    if (queryText.trim()) {
      try {
        const embedded = await this.ai.embed([queryText], "query");
        const vector = embedded.vectors[0];
        if (vector) {
          [chunks, partnerFacets] = await Promise.all([
            this.vectors.searchChunks(conversationId, vector),
            partnerStyle
              ? this.vectors.searchFacets(
                  conversation.otherUserId,
                  partnerStyle.version,
                  vector,
                )
              : Promise.resolve([]),
          ]);
        }
      } catch (error) {
        // 向量服務不可用（沒設定 key、額度用完）時照樣產生推薦，只是沒有 RAG。
        if (!(error instanceof AiError)) throw error;
        console.warn("ai_embed_unavailable", error.code);
      }
    }

    // --- 4. 呼叫 AI 服務 ------------------------------------------------
    const payload: AiSuggestRequest = {
      requestId,
      requester,
      partner,
      sharedTags: this.data.sharedTags(requester, partner),
      recentMessages,
      retrievedChunks: chunks,
      conversationSummary: summary?.summary ?? null,
      requesterStyle: requesterStyle?.card ?? null,
      partnerStyle: partnerStyle?.card ?? null,
      partnerFacets,
      siteStats: await this.data.siteStats(),
      excludeTexts: await this.recentSuggestionTexts(
        conversationId,
        userId,
        lastMessage?.id ?? null,
      ),
    };
    // 沒有風格卡就順手排一次背景萃取（規格 8：需要時發現沒有）；1 小時內只會排一次。
    for (const [card, id] of [
      [requesterStyle, userId],
      [partnerStyle, conversation.otherUserId],
    ] as const)
      if (!card) void this.jobs.enqueueStyle(id).catch(() => undefined);

    let result;
    try {
      result = await this.ai.suggest(payload);
    } catch (error) {
      if (!(error instanceof AiError)) throw error;
      await this.saveFailure(
        requestId,
        conversationId,
        userId,
        lastMessage?.id ?? null,
        error.code,
        Date.now() - started,
      );
      return error.code === "LLM_NOT_CONFIGURED"
        ? fail(503, "AI_NOT_CONFIGURED", "AI 推薦尚未設定，請稍後再試。")
        : fail(503, "AI_UNAVAILABLE", "AI 忙碌中，請稍後再試。");
    }

    // --- 5. 全部存檔（含被刪掉的候選）再回傳 -----------------------------
    const kept = result.suggestions.map((suggestion) => ({
      id: randomUUID(),
      rank: suggestion.rank,
      text: suggestion.text,
      intent: suggestion.intent,
      styleTarget: suggestion.styleTarget,
      styleDistance: suggestion.styleDistance,
      reason: suggestion.reason || null,
      rejectedReason: null,
    }));
    await this.db.aiSuggestionRequest.create({
      data: {
        id: requestId,
        conversationId,
        requesterId: userId,
        mode: result.mode,
        lastMessageId: lastMessage?.id ?? null,
        status: result.status,
        modelName: result.modelName ?? null,
        promptVersion: result.promptVersion,
        inputTokens: result.usage.inputTokens,
        outputTokens: result.usage.outputTokens,
        modelRequests: result.usage.requests,
        latencyMs: Date.now() - started,
        notice: result.notice ?? null,
        suggestions: {
          create: [
            ...kept,
            // rank 0 是被後處理刪掉的候選；決策要求「全部保存到專案結束」。
            ...result.rejected.map((rejected) => ({
              rank: 0,
              text: rejected.text,
              rejectedReason: rejected.reasonCode,
            })),
          ],
        },
      },
    });
    return {
      requestId,
      status: result.status,
      mode: result.mode,
      notice: result.notice ?? null,
      suggestions: kept.map((suggestion) => ({
        id: suggestion.id,
        rank: suggestion.rank,
        text: suggestion.text,
        intent: suggestion.intent,
      })),
    };
  }

  /**
   * 同一個情境下已經給過的推薦，下次要避開（「換一批」）。
   *
   * 只比對「依據同一則最後訊息」產生的推薦：對話往前走之後，舊推薦本來就不會再出現，
   * 不需要排除，也避免把可用的說法永久封鎖。
   */
  private async recentSuggestionTexts(
    conversationId: string,
    requesterId: string,
    lastMessageId: string | null,
  ) {
    const rows = await this.db.aiSuggestion.findMany({
      where: {
        rank: { gt: 0 },
        request: { conversationId, requesterId, lastMessageId },
      },
      orderBy: { createdAt: "desc" },
      take: EXCLUDE_LIMIT,
      select: { text: true },
    });
    return rows.map((row) => row.text);
  }

  /**
   * AI 服務不可用時也留下請求紀錄（規格 9：連失敗的請求都要能評估）。
   * 這裡的寫入失敗不該蓋掉原本的 503，所以錯誤只記錄不往外丟。
   */
  private async saveFailure(
    requestId: string,
    conversationId: string,
    requesterId: string,
    lastMessageId: string | null,
    errorCode: string,
    latencyMs: number,
  ) {
    await this.db.aiSuggestionRequest
      .create({
        data: {
          id: requestId,
          conversationId,
          requesterId,
          // 模式由 AI 服務判斷，失敗時無從得知，用 unknown 佔位。
          mode: "unknown",
          lastMessageId,
          status: "error",
          errorCode,
          latencyMs,
        },
      })
      .catch(() => console.error("ai_request_log_failed"));
  }
}
