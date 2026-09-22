// 訊息來源標記（規格 5.6）。
//
// 使用者送出訊息時可以帶 suggestionId，代表這則訊息是從某個 AI 推薦來的。
// 後端比較「推薦原文」與「實際送出的內容」有多像，決定 origin：
//   ≥ 0.95 ai_verbatim（原封不動）／≥ 0.5 ai_edited（改過再送）／< 0.5 human（等於自己重寫）。
// ai_verbatim 與 ai_edited 在萃取風格卡時會被排除，避免 AI 學 AI 的回饋迴圈。
//
// 這張表刻意與 messages 分開：送出訊息時整筆會經由 message:new 推給聊天對象，
// 來源欄位若放在 messages，對方就會看到「這則是 AI 產生的」。
import { Injectable } from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { fail } from "./core";
import { classifyOrigin, similarity } from "./ai-text";

@Injectable()
export class MessageOrigins {
  /**
   * 在「建立訊息的同一個交易」裡寫下這則訊息的來源。
   *
   * 沒有帶 suggestionId 就什麼都不寫：查詢時用 COALESCE(origin, 'human')，
   * 沒有紀錄就代表真人訊息，可以少寫一張表的資料。
   *
   * 會檢查這則推薦確實是「這個人在這個聊天室」拿到的，避免有人用別人的 suggestionId
   * 把自己的訊息標成 AI（或反過來竄改別人的統計）。
   *
   * 即使相似度低到算 human，仍然保留 suggestionId：評估「A 修改幅度」時要用（規格 12）。
   */
  async record(
    tx: Prisma.TransactionClient,
    input: {
      messageId: string;
      conversationId: string;
      senderId: string;
      content: string;
      suggestionId?: string;
    },
  ) {
    if (!input.suggestionId) return null;
    const suggestion = await tx.aiSuggestion.findFirst({
      where: {
        id: input.suggestionId,
        request: {
          requesterId: input.senderId,
          conversationId: input.conversationId,
        },
      },
      select: { id: true, text: true },
    });
    if (!suggestion)
      return fail(404, "SUGGESTION_NOT_FOUND", "找不到這則 AI 推薦。");
    const score = similarity(suggestion.text, input.content);
    const origin = classifyOrigin(score);
    await tx.messageOrigin.create({
      data: {
        messageId: input.messageId,
        origin,
        similarity: score,
        suggestionId: suggestion.id,
      },
    });
    // 記下「這則推薦被採用了」，之後才能算第 1 則的採用率。
    await tx.aiSuggestion.update({
      where: { id: suggestion.id },
      data: { chosenAt: new Date() },
    });
    return { origin, similarity: score };
  }
}
