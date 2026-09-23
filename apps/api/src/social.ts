import { Injectable } from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { z } from "zod";
import {
  Database,
  fail,
  parse,
  uuid,
  ageAt,
  distance,
  pairLock,
  Infrastructure,
} from "./core";
import {
  Profiles,
  card,
  preferenceHeightRange,
  traitCodes,
  userInclude,
} from "./profiles";
import { MessageOrigins } from "./ai-origins";
import { AiJobs } from "./ai-jobs";
const messageInput = z
  .object({
    content: z.string().trim().min(1).max(2000),
    clientId: z.string().uuid(),
    // 這則訊息是從哪個 AI 推薦來的（選填）；後端據此標記來源（規格 5.6）。
    suggestionId: z.string().uuid().optional(),
  })
  .strict();
@Injectable()
export class Social {
  constructor(
    private db: Database,
    private profiles: Profiles,
    private infra: Infrastructure,
    private origins: MessageOrigins,
    private jobs: AiJobs,
  ) {}
  publish: (userId: string, event: string, data: unknown) => void = () => {};
  revoke: (conversationId: string) => void = () => {};
  eligible(a: any, b: any) {
    if (!a?.profile || !b?.profile || !a.preference || !b.preference)
      return false;
    const p = a.preference,
      q = b.profile;
    return (
      ageAt(q.birthDate) >= p.minAge &&
      ageAt(q.birthDate) <= p.maxAge &&
      (p.preferredGender === "any" || p.preferredGender === q.gender) &&
      // 關係期待改看 traits 的 dating_goal：偏好「都可以」，或對方的交友目標包含它。
      (p.preferredDatingIntent === "any" ||
        traitCodes(b, true).includes(p.preferredDatingIntent)) &&
      // 沒填身高的人無從判斷，不因身高條件被排除。拉桿停在兩端代表不限：
      // 身高必填後，低於 130 的人沒辦法留空，不能讓預設偏好把他們擋掉。
      (q.heightCm == null ||
        ((p.minHeightCm <= preferenceHeightRange[0] ||
          q.heightCm >= p.minHeightCm) &&
          (p.maxHeightCm >= preferenceHeightRange[1] ||
            q.heightCm <= p.maxHeightCm))) &&
      distance(a.profile, q) <= p.maxDistanceKm
    );
  }
  async discovery(id: string) {
    const me = await this.db.user.findUniqueOrThrow({
      where: { id },
      include: userInclude,
    });
    if (!me.profile) return [];
    const excluded = await this.db.interaction.findMany({
      where: { fromUserId: id },
      select: { toUserId: true },
    });
    const users = await this.db.user.findMany({
      where: {
        id: { notIn: [id, ...excluded.map((x) => x.toUserId)] },
        profile: { isNot: null },
        blocks: { none: { blockedUserId: id } },
        blockedBy: { none: { userId: id } },
        matchesA: { none: { userBId: id } },
        matchesB: { none: { userAId: id } },
      },
      include: userInclude,
      // 匯入資料的 createdAt 完全相同，加上 id 當第二排序鍵才有穩定順序。
      orderBy: [{ createdAt: "desc" }, { id: "asc" }],
      take: 500,
    });
    return users
      .filter((u) => this.eligible(me, u) && this.eligible(u, me))
      .slice(0, 30)
      .map((u) => card(u, id));
  }
  /**
   * 測試用：用顯示名稱或 email 搜尋除了自己以外的全部使用者（不分大小寫、部分符合），
   * 已經按過喜歡／略過的人也列出，可以重新按來改變狀態。沒有個人檔案的人畫不出卡片，不列出。
   * 每筆附上 searchState：我按過什麼、配對狀態、雙方偏好是否相符、是否封鎖中。
   * 從搜尋列按喜歡／略過走 POST /discovery/search/interactions（interact 的測試模式）。
   */
  async search(id: string, q: unknown) {
    const text = parse(z.string().trim().min(1).max(100), q);
    const me = await this.db.user.findUniqueOrThrow({
      where: { id },
      include: userInclude,
    });
    const users = await this.db.user.findMany({
      where: {
        id: { not: id },
        profile: { isNot: null },
        OR: [
          { email: { contains: text, mode: "insensitive" } },
          { profile: { displayName: { contains: text, mode: "insensitive" } } },
        ],
      },
      include: {
        ...userInclude,
        receivedInteractions: {
          where: { fromUserId: id },
          select: { action: true },
        },
        blocks: { where: { blockedUserId: id }, select: { id: true } },
        blockedBy: { where: { userId: id }, select: { id: true } },
        matchesA: { where: { userBId: id }, select: { status: true } },
        matchesB: { where: { userAId: id }, select: { status: true } },
      },
      orderBy: [{ createdAt: "desc" }, { id: "asc" }],
      take: 20,
    });
    return users.map((u) => {
      const match = [...u.matchesA, ...u.matchesB][0];
      const blocked = u.blocks.length > 0 || u.blockedBy.length > 0;
      const view = card(u, id)!;
      return {
        ...view,
        // 封鎖後照片連結會回 404，乾脆不給。
        photos: blocked ? [] : view.photos,
        searchState: {
          action: u.receivedInteractions[0]?.action ?? null,
          match: match
            ? match.status === "active"
              ? "active"
              : "ended"
            : null,
          eligible: this.eligible(me, u) && this.eligible(u, me),
          blocked,
        },
      };
    });
  }
  /**
   * 按喜歡／略過。test 為 true 時是搜尋列（測試用）送來的，規則放寬：
   * ・封鎖中會先解除雙方的封鎖（兩個方向都解除），再照常處理。
   * ・不檢查雙方偏好。
   * ・已配對時按略過會解除配對（同 unmatch）；配對已結束時互相喜歡，恢復原本的配對。
   */
  async interact(id: string, body: unknown, test = false) {
    const dto = parse(
      z
        .object({
          targetUserId: z.string().uuid(),
          action: z.enum(["like", "pass"]),
        })
        .strict(),
      body,
    );
    if (id === dto.targetUserId)
      return fail(400, "SELF_INTERACTION", "無法對自己操作。");
    // 測試模式解除配對時，交易完成後要斷開這個聊天室的即時連線。
    let revoked: string | undefined;
    const result = await this.db.$transaction(async (tx) => {
      await pairLock(tx, id, dto.targetUserId);
      const pairBlocks = {
        OR: [
          { userId: id, blockedUserId: dto.targetUserId },
          { userId: dto.targetUserId, blockedUserId: id },
        ],
      };
      if (test) await tx.block.deleteMany({ where: pairBlocks });
      else if (await tx.block.findFirst({ where: pairBlocks }))
        return fail(403, "BLOCKED", "目前無法互動。");
      const users = await tx.user.findMany({
        where: { id: { in: [id, dto.targetUserId] } },
        include: userInclude,
      });
      const me = users.find((u) => u.id === id),
        other = users.find((u) => u.id === dto.targetUserId);
      if (!other) return fail(404, "NOT_FOUND", "找不到使用者。");
      if (
        !test &&
        (!me || !this.eligible(me, other) || !this.eligible(other, me))
      )
        return fail(409, "NOT_ELIGIBLE", "對方目前不符合雙方偏好。");
      const [userAId, userBId] = [id, dto.targetUserId].sort();
      const existing = await tx.match.findUnique({
        where: { userAId_userBId: { userAId, userBId } },
        include: { conversation: true },
      });
      if (existing && !test)
        return {
          matched: existing.status === "active",
          matchId: existing.status === "active" ? existing.id : undefined,
          created: false,
        };
      await tx.interaction.upsert({
        where: {
          fromUserId_toUserId: { fromUserId: id, toUserId: dto.targetUserId },
        },
        create: {
          fromUserId: id,
          toUserId: dto.targetUserId,
          action: dto.action,
        },
        update: { action: dto.action },
      });
      // 以下只有測試模式會遇到：既有配對。
      if (existing?.status === "active") {
        if (dto.action === "like")
          return { matched: true, matchId: existing.id, created: false };
        await tx.match.update({
          where: { id: existing.id },
          data: { status: "unmatched", unmatchedAt: new Date() },
        });
        revoked = existing.conversation?.id;
        return { matched: false, created: false };
      }
      const reciprocal = await tx.interaction.findUnique({
        where: {
          fromUserId_toUserId: { fromUserId: dto.targetUserId, toUserId: id },
        },
      });
      if (dto.action !== "like" || reciprocal?.action !== "like")
        return { matched: false, created: false };
      // 配對已結束（只有測試模式會走到這裡）：恢復原本的配對，聊天紀錄一併回來。
      const match = existing
        ? await tx.match.update({
            where: { id: existing.id },
            data: { status: "active", unmatchedAt: null },
            include: { conversation: true },
          })
        : await tx.match.create({
            data: {
              userAId,
              userBId,
              conversation: {
                create: {
                  members: {
                    create: [{ userId: id }, { userId: dto.targetUserId }],
                  },
                },
              },
            },
            include: { conversation: true },
          });
      for (const userId of [id, dto.targetUserId])
        await tx.notification.create({
          data: {
            userId,
            type: "match",
            payload: {
              matchId: match.id,
              conversationId: match.conversation!.id,
            },
          },
        });
      return { matched: true, matchId: match.id, created: true };
    });
    if (revoked) this.revoke(revoked);
    if (result.created)
      for (const userId of [id, dto.targetUserId])
        this.publish(userId, "notification:new", {
          type: "match",
          matchId: result.matchId,
        });
    return {
      matched: result.matched,
      ...(result.matchId ? { matchId: result.matchId } : {}),
    };
  }
  /**
   * 我按過「喜歡」的人（送出的喜歡）。
   * status：waiting＝還在等對方回應；matched＝對方也喜歡你，conversationId 就是聊天室。
   *
   * 可見性條件直接寫在查詢裡（同 discovery 的做法），take 才會是「可見的前 200 筆」，
   * 而不是先抓 200 筆再濾掉一堆、讓實際筆數因人而異：
   * ・任一方封鎖、或對方還沒填個人檔案 → 不列出（與配對清單同一套規則）。
   * ・配對已結束（unmatched／blocked）→ 不列出：interact() 遇到既有配對就不會再寫 like，
   *   這種人留著只會永遠顯示「等待回應」。
   */
  async likesSent(id: string) {
    const rows = await this.db.interaction.findMany({
      where: {
        fromUserId: id,
        action: "like",
        toUser: {
          profile: { isNot: null },
          blocks: { none: { blockedUserId: id } },
          blockedBy: { none: { userId: id } },
          matchesA: { none: { userBId: id, status: { not: "active" } } },
          matchesB: { none: { userAId: id, status: { not: "active" } } },
        },
      },
      include: { toUser: { include: userInclude } },
      orderBy: { createdAt: "desc" },
      take: 200,
    });
    const matches = await this.db.match.findMany({
      where: { status: "active", OR: [{ userAId: id }, { userBId: id }] },
      include: { conversation: { select: { id: true } } },
    });
    const matched = new Map(
      matches.map((m) => [m.userAId === id ? m.userBId : m.userAId, m]),
    );
    return rows.map((row) => {
      const m = matched.get(row.toUserId);
      return {
        targetUserId: row.toUserId,
        createdAt: row.createdAt,
        status: m ? "matched" : "waiting",
        ...(m ? { matchId: m.id } : {}),
        ...(m?.conversation ? { conversationId: m.conversation.id } : {}),
        user: card(row.toUser, id),
      };
    });
  }
  async matches(id: string, matchId?: string) {
    if (matchId) uuid(matchId);
    const matches = await this.db.match.findMany({
      where: {
        ...(matchId ? { id: matchId } : {}),
        status: "active",
        OR: [{ userAId: id }, { userBId: id }],
      },
      include: {
        userA: { include: userInclude },
        userB: { include: userInclude },
        conversation: true,
      },
      orderBy: { createdAt: "desc" },
    });
    const result = [];
    for (const m of matches) {
      const other = m.userAId === id ? m.userB : m.userA;
      if (!(await this.profiles.blocked(id, other.id)))
        result.push({
          id: m.id,
          createdAt: m.createdAt,
          otherUser: card(other, id),
          conversationId: m.conversation?.id,
        });
    }
    if (matchId && !result[0]) return fail(404, "NOT_FOUND", "找不到配對。");
    return matchId ? result[0] : result;
  }
  async unmatch(id: string, matchId: string) {
    uuid(matchId);
    const m = await this.db.match.findFirst({
      where: { id: matchId, OR: [{ userAId: id }, { userBId: id }] },
      include: { conversation: true },
    });
    if (!m) return fail(404, "NOT_FOUND", "找不到配對。");
    await this.db.$transaction(async (tx) => {
      await pairLock(tx, m.userAId, m.userBId);
      await tx.match.update({
        where: { id: matchId },
        data: { status: "unmatched", unmatchedAt: new Date() },
      });
    });
    if (m.conversation) this.revoke(m.conversation.id);
    return { ok: true };
  }
  async block(id: string, target: string) {
    uuid(target);
    if (id === target) return fail(400, "SELF_BLOCK", "無法封鎖自己。");
    if (!(await this.db.user.findUnique({ where: { id: target } })))
      return fail(404, "NOT_FOUND", "找不到使用者。");
    const [userAId, userBId] = [id, target].sort();
    const conversation = await this.db.$transaction(async (tx) => {
      await pairLock(tx, id, target);
      await tx.block.upsert({
        where: { userId_blockedUserId: { userId: id, blockedUserId: target } },
        create: { userId: id, blockedUserId: target },
        update: {},
      });
      const m = await tx.match.findUnique({
        where: { userAId_userBId: { userAId, userBId } },
        include: { conversation: true },
      });
      if (m)
        await tx.match.update({
          where: { id: m.id },
          data: { status: "blocked", unmatchedAt: new Date() },
        });
      return m?.conversation;
    });
    if (conversation) this.revoke(conversation.id);
    return { ok: true };
  }
  async unblock(id: string, target: string) {
    uuid(target);
    await this.db.block.deleteMany({
      where: { userId: id, blockedUserId: target },
    });
    return { ok: true };
  }
  async blocks(id: string) {
    return (
      await this.db.block.findMany({
        where: { userId: id },
        include: { blockedUser: { include: { profile: true } } },
      })
    ).map((b) => ({
      blockedUserId: b.blockedUserId,
      displayName: b.blockedUser.profile?.displayName || "使用者",
    }));
  }
  async access(
    id: string,
    conversationId: string,
    tx: Prisma.TransactionClient = this.db,
  ) {
    uuid(conversationId);
    const c = await tx.conversation.findUnique({
      where: { id: conversationId },
      include: { match: true },
    });
    if (!c || ![c.match.userAId, c.match.userBId].includes(id))
      return fail(404, "NOT_FOUND", "找不到聊天室。");
    if (c.match.status !== "active")
      return fail(403, "CONVERSATION_CLOSED", "這段對話已結束。");
    const other = c.match.userAId === id ? c.match.userBId : c.match.userAId;
    if (
      await tx.block.findFirst({
        where: {
          OR: [
            { userId: id, blockedUserId: other },
            { userId: other, blockedUserId: id },
          ],
        },
      })
    )
      return fail(403, "BLOCKED", "目前無法傳送訊息。");
    return { ...c, otherUserId: other };
  }
  async conversations(id: string) {
    const rows = await this.db.conversation.findMany({
      where: { members: { some: { userId: id } }, match: { status: "active" } },
      include: {
        members: true,
        messages: { orderBy: { createdAt: "desc" }, take: 1 },
        match: {
          include: {
            userA: { include: userInclude },
            userB: { include: userInclude },
          },
        },
      },
      orderBy: { updatedAt: "desc" },
    });
    const out = [];
    for (const c of rows) {
      const other = c.match.userAId === id ? c.match.userB : c.match.userA;
      if (await this.profiles.blocked(id, other.id)) continue;
      const lastReadAt = c.members.find((m) => m.userId === id)?.lastReadAt;
      out.push({
        id: c.id,
        matchId: c.matchId,
        otherUser: card(other, id),
        lastMessage: c.messages[0] || null,
        otherLastReadAt:
          c.members.find((member) => member.userId === other.id)?.lastReadAt ??
          null,
        unreadCount: await this.db.message.count({
          where: {
            conversationId: c.id,
            senderId: { not: id },
            ...(lastReadAt ? { createdAt: { gt: lastReadAt } } : {}),
          },
        }),
      });
    }
    return out;
  }
  /**
   * 這位使用者所有有效聊天室的對象（聊天室 id ＋ 對方的 id）。
   * 即時上線狀態要用：連線／斷線時要通知每一位對象，
   * 對話列表也要能一次問出「我的這些對象現在誰在線上」。
   */
  async activePartners(id: string) {
    const rows = await this.db.conversation.findMany({
      where: { members: { some: { userId: id } }, match: { status: "active" } },
      select: { id: true, match: { select: { userAId: true, userBId: true } } },
    });
    return rows.map((row) => ({
      conversationId: row.id,
      otherUserId:
        row.match.userAId === id ? row.match.userBId : row.match.userAId,
    }));
  }
  async conversation(id: string, conversationId: string) {
    await this.access(id, conversationId);
    return (await this.conversations(id)).find((c) => c.id === conversationId);
  }
  async createConversation(id: string, body: unknown) {
    const { matchId } = parse(
      z.object({ matchId: z.string().uuid() }).strict(),
      body,
    );
    const m = (await this.matches(id, matchId)) as any;
    return this.conversation(id, m.conversationId);
  }
  async messages(
    id: string,
    conversationId: string,
    before?: string,
    beforeId?: string,
  ) {
    await this.access(id, conversationId);
    if (before) parse(z.iso.datetime("時間格式不正確"), before);
    if (beforeId) {
      if (!before) return fail(400, "VALIDATION_ERROR", "分頁參數不完整");
      uuid(beforeId);
    }
    const at = before ? new Date(before) : undefined;
    const messages = await this.db.message.findMany({
      where: {
        conversationId,
        ...(at
          ? beforeId
            ? {
                OR: [
                  { createdAt: { lt: at } },
                  { createdAt: at, id: { lt: beforeId } },
                ],
              }
            : { createdAt: { lt: at } }
          : {}),
      },
      orderBy: [{ createdAt: "desc" }, { id: "desc" }],
      take: 50,
    });
    return messages.reverse();
  }
  async send(id: string, conversationId: string, body: unknown) {
    // suggestionId 不是 messages 的欄位，拆出來單獨處理（寫進 message_origins）。
    const { suggestionId, ...data } = parse(messageInput, body);
    await this.infra.limit(`message:${id}`, 60);
    const initial = await this.access(id, conversationId);
    const out = await this.db.$transaction(async (tx) => {
      await pairLock(tx, id, initial.otherUserId);
      const c = await this.access(id, conversationId, tx);
      const existing = await tx.message.findUnique({
        where: { senderId_clientId: { senderId: id, clientId: data.clientId } },
      });
      if (existing) {
        if (
          existing.conversationId !== conversationId ||
          existing.content !== data.content
        )
          return fail(
            409,
            "IDEMPOTENCY_CONFLICT",
            "訊息識別碼已用於其他內容。",
          );
        return { message: existing, created: false, other: c.otherUserId };
      }
      const message = await tx.message.create({
        data: { ...data, senderId: id, conversationId },
      });
      // 來源標記與訊息同一個交易：不會出現「訊息存了、來源沒存」的狀態（規格 5.6）。
      await this.origins.record(tx, {
        messageId: message.id,
        conversationId,
        senderId: id,
        content: message.content,
        suggestionId,
      });
      await tx.conversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() },
      });
      await tx.notification.create({
        data: {
          userId: c.otherUserId,
          type: "message",
          payload: { conversationId, messageId: message.id },
        },
      });
      return { message, created: true, other: c.otherUserId };
    });
    if (out.created) {
      for (const user of [id, out.other])
        this.publish(user, "message:new", out.message);
      this.publish(out.other, "notification:new", {
        type: "message",
        conversationId,
      });
      // 背景 AI 工作（切片、話題區段、摘要、風格卡）不擋回應：
      // 排程失敗只影響之後的推薦品質，不該讓使用者送不出訊息。
      void this.jobs
        .afterMessage(conversationId, id)
        .catch(() => console.error("ai_jobs_enqueue_failed"));
    }
    return out.message;
  }
  async read(id: string, conversationId: string) {
    const c = await this.access(id, conversationId);
    const at = new Date();
    await this.db.conversationMember.update({
      where: { conversationId_userId: { conversationId, userId: id } },
      data: { lastReadAt: at },
    });
    this.publish(c.otherUserId, "conversation:read", {
      conversationId,
      userId: id,
      readAt: at.toISOString(),
    });
    return { ok: true };
  }
  async notifications(id: string) {
    return this.db.notification.findMany({
      where: { userId: id },
      orderBy: { createdAt: "desc" },
      take: 100,
    });
  }
  async readNotification(id: string, notificationId: string) {
    uuid(notificationId);
    const updated = await this.db.notification.updateMany({
      where: { id: notificationId, userId: id },
      data: { readAt: new Date() },
    });
    if (!updated.count) return fail(404, "NOT_FOUND", "找不到通知。");
    return { ok: true };
  }
}
