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
import { Profiles, card, userInclude } from "./profiles";
const messageInput = z
  .object({
    content: z.string().trim().min(1).max(2000),
    clientId: z.string().uuid(),
  })
  .strict();
@Injectable()
export class Social {
  constructor(
    private db: Database,
    private profiles: Profiles,
    private infra: Infrastructure,
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
      (p.preferredDatingIntent === "any" ||
        p.preferredDatingIntent === q.datingIntent) &&
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
      orderBy: { createdAt: "desc" },
      take: 500,
    });
    return users
      .filter((u) => this.eligible(me, u) && this.eligible(u, me))
      .slice(0, 30)
      .map(card);
  }
  async interact(id: string, body: unknown) {
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
    const result = await this.db.$transaction(async (tx) => {
      await pairLock(tx, id, dto.targetUserId);
      if (
        await tx.block.findFirst({
          where: {
            OR: [
              { userId: id, blockedUserId: dto.targetUserId },
              { userId: dto.targetUserId, blockedUserId: id },
            ],
          },
        })
      )
        return fail(403, "BLOCKED", "目前無法互動。");
      const users = await tx.user.findMany({
        where: { id: { in: [id, dto.targetUserId] } },
        include: userInclude,
      });
      const me = users.find((u) => u.id === id),
        other = users.find((u) => u.id === dto.targetUserId);
      if (!other) return fail(404, "NOT_FOUND", "找不到使用者。");
      if (!me || !this.eligible(me, other) || !this.eligible(other, me))
        return fail(409, "NOT_ELIGIBLE", "對方目前不符合雙方偏好。");
      const [userAId, userBId] = [id, dto.targetUserId].sort();
      const existing = await tx.match.findUnique({
        where: { userAId_userBId: { userAId, userBId } },
      });
      if (existing)
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
      const reciprocal = await tx.interaction.findUnique({
        where: {
          fromUserId_toUserId: { fromUserId: dto.targetUserId, toUserId: id },
        },
      });
      if (dto.action !== "like" || reciprocal?.action !== "like")
        return { matched: false, created: false };
      const match = await tx.match.create({
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
          otherUser: card(other),
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
        otherUser: card(other),
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
  async activeConversationIds(id: string) {
    const rows = await this.db.conversation.findMany({
      where: { members: { some: { userId: id } }, match: { status: "active" } },
      select: { id: true },
    });
    return rows.map((c) => c.id);
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
    const data = parse(messageInput, body);
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
