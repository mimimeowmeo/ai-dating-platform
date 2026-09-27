import { Injectable } from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { z } from "zod";
import {
  Database,
  fail,
  parse,
  uuid,
  ageAt,
  pairLock,
  Infrastructure,
} from "./core";
import { Profiles, card, photoView, userInclude } from "./profiles";
import { eligible, eligibleIds } from "./eligibility";
import { MessageOrigins } from "./ai-origins";
import { AiJobs } from "./ai-jobs";
import { VectorStore } from "./ai-store";
import {
  NEAR_THRESHOLDS,
  RANKING_VERSION,
  buildQueue,
  explainInterest,
  passPenalty,
  scoreCandidates,
} from "./discovery-rank";
import { ServedCard, SwipeLogs } from "./swipe-logs";
import { RecJobs } from "./rec-jobs";
/**
 * 探索頁的開關（測試用，網址參數見 docs/testing/QUERY-PARAMS.md）。
 * hardFilter：從全部使用者篩出雙向的年齡、性別、交往目的、身高、距離都符合的人（見 eligibility.ts）。
 * 封鎖、已按過、已配對的人不管開關都排除。appearance／interest：排序用外貌、個人標籤分數；explain：附上推薦依據。
 */
export type DiscoveryOptions = {
  hardFilter: boolean;
  appearance: boolean;
  interest: boolean;
  explain: boolean;
};
const allDiscoveryOn: DiscoveryOptions = {
  hardFilter: true,
  appearance: true,
  interest: true,
  explain: false,
};
/** 排序可用的分數：face 外貌、tags 個人標籤；rank 參數可以多選，預設全部。 */
export const RANK_SIGNALS = ["face", "tags"] as const;
/** 2026-09-28 改名前的參數；帶了舊名字就回 400，告訴對方新寫法。 */
const renamedParams: Record<string, string> = {
  hardfilter: "hardfilter=false 改成 prefs=off",
  appearance: "appearance=false 改成 rank=tags",
  interest: "interest=false 改成 rank=face",
  test: "test=true 改成 debug=explain",
};
/** 逗號分隔或重複的參數（rank=face,tags 或 rank=face&rank=tags）都轉成陣列。 */
const commaList = (value: unknown) =>
  value === undefined
    ? undefined
    : (Array.isArray(value) ? value : [value])
        .flatMap((v) => String(v).split(","))
        .map((v) => v.trim())
        .filter(Boolean);
const discoveryQuery = z.strictObject({
  prefs: z.enum(["on", "off"], "prefs 只能是 on 或 off").default("on"),
  rank: z.preprocess(
    commaList,
    z
      .array(z.enum(RANK_SIGNALS, "rank 只能是 face、tags（可用逗號多選）"))
      .min(1, "rank 至少要選一個：face、tags")
      .default([...RANK_SIGNALS]),
  ),
  // search 只影響前端（改顯示測試用搜尋列），API 收到不做事。
  debug: z.preprocess(
    commaList,
    z
      .array(
        z.enum(
          ["explain", "search"],
          "debug 只能是 explain、search（可用逗號多選）",
        ),
      )
      .default([]),
  ),
});
/** GET /discovery 的網址參數 → 探索開關；參數名稱或值不認得就回 400。 */
export function discoveryOptions(query: unknown): DiscoveryOptions {
  const input = (query ?? {}) as Record<string, unknown>;
  const renamed = Object.keys(input)
    .filter((key) => key in renamedParams)
    .map((key) => renamedParams[key]);
  if (renamed.length)
    return fail(
      400,
      "VALIDATION_ERROR",
      `網址參數已改名：${renamed.join("；")}`,
    );
  const parsed = discoveryQuery.safeParse(input);
  if (!parsed.success)
    return fail(
      400,
      "VALIDATION_ERROR",
      [
        ...new Set(
          parsed.error.issues.map((issue) =>
            issue.code === "unrecognized_keys"
              ? `不認得的參數：${issue.keys.join("、")}（可用 prefs、rank、debug）`
              : issue.message,
          ),
        ),
      ].join("；"),
    );
  const { prefs, rank, debug } = parsed.data;
  return {
    hardFilter: prefs === "on",
    appearance: rank.includes("face"),
    interest: rank.includes("tags"),
    explain: debug.includes("explain"),
  };
}
type RankedCard = { userId: string; served: ServedCard };
/** 外貌分數用最近幾個「喜歡」當參考臉。 */
const APPEARANCE_ANCHORS = 20;
/** 略過扣分最多看最近幾個「略過」：Core V1 看全部，時間一久扣分會越積越多。 */
const PASS_HISTORY = 200;
/** 探索頁一次回傳幾張卡。 */
const DISCOVERY_LIMIT = 30;
/** 「我的配對」看得到的喜歡過的人（和 likesSent 相同）：有個人檔案、雙方都沒封鎖、配對沒有結束。 */
const visibleLiked = (id: string): Prisma.UserWhereInput => ({
  profile: { isNot: null },
  blocks: { none: { blockedUserId: id } },
  blockedBy: { none: { userId: id } },
  matchesA: { none: { userBId: id, status: { not: "active" } } },
  matchesB: { none: { userAId: id, status: { not: "active" } } },
});
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
    private vectors: VectorStore,
    private swipeLogs: SwipeLogs,
    private rec: RecJobs,
  ) {}
  publish: (userId: string, event: string, data: unknown) => void = () => {};
  revoke: (conversationId: string) => void = () => {};
  async discovery(id: string, options: DiscoveryOptions = allDiscoveryOn) {
    const me = await this.db.user.findUniqueOrThrow({
      where: { id },
      include: userInclude,
    });
    if (!me.profile) return [];
    // 候選池不限註冊時間：硬篩選開著時是全部使用者裡雙方都符合偏好的人，關掉時是整個使用者池。
    // 兩種都先只取 id 排序，最後 30 人才載入卡片資料。
    const pool = options.hardFilter
      ? await eligibleIds(this.db, id)
      : await this.everyoneIds(id);
    const ranked = await this.rank(id, pool, options);
    const users = await this.db.user.findMany({
      where: { id: { in: ranked.map((r) => r.userId) } },
      include: userInclude,
    });
    const byId = new Map(users.map((u) => [u.id, u]));
    const shown = ranked.filter((r) => byId.has(r.userId));
    await this.swipeLogs.remember(id, shown);
    const explain = options.explain
      ? await this.explainer(me, shown, byId)
      : null;
    return shown.map((r) => {
      const view = card(byId.get(r.userId), id);
      return explain ? { ...view, explain: explain(r) } : view;
    });
  }
  /**
   * 關掉硬篩選時的候選池：除了自己、按過喜歡／略過、任一方封鎖、配對過的人以外的全部使用者，
   * 最新註冊在前（匯入資料的 createdAt 完全相同，再依 id）。
   */
  private async everyoneIds(id: string) {
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
      select: { id: true },
      orderBy: [{ createdAt: "desc" }, { id: "asc" }],
    });
    return users.map((u) => u.id);
  }
  /**
   * AI 排序，回傳最多 30 人，每張卡附上當下的推薦狀態（寫滑卡紀錄、測試畫面用）：
   * 1. 外貌分數 = 和我最近按喜歡的 20 人最像的相似度，再扣 PASS V2（附近被我略過的人太多就扣，
   *    幅度依種子分布校準，女最多約 0.034、男約 0.051）。
   * 2. 外貌、興趣分數各自換成候選池裡的百分位再相加：都開時外貌 60%、興趣 40%，關掉其中一個另一個就是 100%。
   *    還沒按過喜歡、或按過的人都沒有外貌向量時，外貌算不出來，改成興趣 100%；候選人沒有外貌向量時給中間值。
   * 3. 每 5 張推薦卡之後插 1 張未推薦卡（見 discovery-rank.ts 的 buildQueue），位置依已滑張數決定。
   *    未推薦卡的種子含日期與喜歡數，同一天沒有新的喜歡時挑到的人不變。
   * 網址參數 rank 至少選一種分數；程式內部呼叫時兩個都關，就維持傳進來的順序（最新註冊在前），不插未推薦卡。
   * 排序失敗不能讓探索頁壞掉，出錯就退回原本順序（source 記成 latest）。
   */
  private async rank(
    id: string,
    ids: string[],
    options: DiscoveryOptions,
  ): Promise<RankedCard[]> {
    if (!ids.length) return [];
    const common = {
      poolSize: ids.length,
      hardFilter: options.hardFilter,
      appearanceOn: options.appearance,
      interestOn: options.interest,
      rankingVersion: RANKING_VERSION,
      servedAt: new Date().toISOString(),
    };
    const latest = () =>
      ids.slice(0, DISCOVERY_LIMIT).map((userId, i) => ({
        userId,
        served: {
          ...common,
          source: "latest" as const,
          position: i + 1,
          rank: i + 1,
          score: null,
          appearanceWeight: null,
          appearancePercentile: null,
          appearanceSimilarity: null,
          passPenalty: null,
          nearLike: null,
          nearPass: null,
          anchorId: null,
          interestPercentile: null,
          interestScore: null,
          likeCount: null,
          swipeCount: null,
        },
      }));
    if (!options.appearance && !options.interest) return latest();
    try {
      const [likeCount, swipeCount] = await Promise.all([
        this.db.interaction.count({
          where: { fromUserId: id, action: "like" },
        }),
        this.db.interaction.count({ where: { fromUserId: id } }),
      ]);
      let appearance: Map<string, number> | null = null;
      const details = new Map<
        string,
        {
          base: number;
          nearLike: number;
          nearPass: number;
          anchorId: string;
          penalty: number;
        }
      >();
      const penalized = new Set<string>();
      if (options.appearance) {
        const [likes, passes] = await Promise.all([
          this.db.interaction.findMany({
            where: { fromUserId: id, action: "like" },
            orderBy: { createdAt: "desc" },
            take: APPEARANCE_ANCHORS,
            select: { toUserId: true },
          }),
          this.db.interaction.findMany({
            where: { fromUserId: id, action: "pass" },
            orderBy: { createdAt: "desc" },
            take: PASS_HISTORY,
            select: { toUserId: true },
          }),
        ]);
        const scores = await this.vectors.appearanceScores(
          likes.map((like) => like.toUserId),
          passes.map((pass) => pass.toUserId),
          ids,
          NEAR_THRESHOLDS,
        );
        appearance = new Map();
        for (const [userId, score] of scores) {
          const penalty = passPenalty(
            score.nearPass,
            score.nearLike,
            score.gender,
          );
          if (penalty > 0) penalized.add(userId);
          appearance.set(userId, Math.max(0, score.base - penalty));
          details.set(userId, { ...score, penalty });
        }
      }
      const interest = options.interest
        ? await this.vectors.interestScores(id, ids)
        : null;
      const scored = scoreCandidates(ids, appearance, interest);
      const rankOf = new Map(scored.ranked.map((userId, i) => [userId, i + 1]));
      const queue = buildQueue(scored.ranked, {
        limit: DISCOVERY_LIMIT,
        swipeCount,
        seed: `${id}:${new Date().toISOString().slice(0, 10)}:${likeCount}`,
        penalized,
      });
      return queue.map(({ userId, source }, i) => {
        const detail = details.get(userId);
        return {
          userId,
          served: {
            ...common,
            source,
            position: i + 1,
            rank: rankOf.get(userId)!,
            score: scored.score?.get(userId) ?? null,
            appearanceWeight: scored.appearanceWeight,
            appearancePercentile: scored.appearancePct?.get(userId) ?? null,
            appearanceSimilarity: detail?.base ?? null,
            passPenalty: detail?.penalty ?? null,
            nearLike: detail?.nearLike ?? null,
            nearPass: detail?.nearPass ?? null,
            anchorId: detail?.anchorId ?? null,
            interestPercentile: scored.interestPct?.get(userId) ?? null,
            interestScore: interest ? (interest.get(userId) ?? 0) : null,
            likeCount,
            swipeCount,
          },
        };
      });
    } catch (error) {
      console.error("DISCOVERY_RANK_FAILED", (error as Error).message);
      return latest();
    }
  }
  /**
   * 測試畫面（?debug=explain）的推薦依據：推薦卡還是未推薦卡、各項分數、共同的標籤、最像哪位喜歡過的人。
   * 喜歡過的人本來就列在「我的配對」裡，這裡只多顯示名字和主照片。
   */
  private async explainer(
    me: { id: string; traits: { trait: { category: string; code: string } }[] },
    cards: RankedCard[],
    byId: Map<
      string,
      { traits: { trait: { category: string; code: string } }[] }
    >,
  ) {
    const anchorIds = [
      ...new Set(
        cards.flatMap((c) => (c.served.anchorId ? [c.served.anchorId] : [])),
      ),
    ];
    const anchors = new Map(
      (
        await this.db.user.findMany({
          where: { id: { in: anchorIds }, ...visibleLiked(me.id) },
          include: {
            profile: true,
            photos: { where: { deletedAt: null, isAvatar: true }, take: 1 },
          },
        })
      ).map((u) => [u.id, u]),
    );
    const tags = (user: {
      traits: { trait: { category: string; code: string } }[];
    }) =>
      user.traits.map((t) => ({
        category: t.trait.category,
        code: t.trait.code,
      }));
    const mine = tags(me);
    const recommendedCount = cards.filter(
      (c) => c.served.source === "recommended",
    ).length;
    return ({ userId, served }: RankedCard) => {
      const categories = served.interestOn
        ? explainInterest(mine, tags(byId.get(userId)!))
        : null;
      const anchor = served.anchorId ? anchors.get(served.anchorId) : null;
      return {
        source: served.source,
        position: served.position,
        rank: served.rank,
        poolSize: served.poolSize,
        recommendedCount,
        score: served.score,
        appearanceWeight: served.appearanceWeight,
        appearance: served.appearanceOn
          ? {
              percentile: served.appearancePercentile,
              similarity: served.appearanceSimilarity,
              penalty: served.passPenalty,
              nearLike: served.nearLike,
              nearPass: served.nearPass,
              anchor: anchor
                ? {
                    userId: anchor.id,
                    displayName: anchor.profile?.displayName ?? "",
                    age: anchor.profile
                      ? ageAt(anchor.profile.birthDate)
                      : null,
                    gender: anchor.profile?.gender ?? "",
                    city: anchor.profile?.city ?? "",
                    photoUrl: anchor.photos[0]
                      ? photoView(anchor.photos[0], me.id).url
                      : null,
                  }
                : null,
            }
          : null,
        interest: categories
          ? {
              percentile: served.interestPercentile,
              score: served.interestScore,
              categories,
            }
          : null,
        sharedTags: categories ? categories.flatMap((c) => c.shared) : [],
      };
    };
  }
  /**
   * 測試畫面「像在哪裡」（GET /discovery/explain-appearance）：只能問上一次探索清單裡的卡，
   * 比的是那張卡當時「最像你喜歡過的人」，不能任意指定兩個人。對方封鎖、配對已結束等情況一律當作找不到。
   * 回傳兩張主照片的網址（框的座標以這兩張為準）與 worker 的狀態：pending 時前端每秒再問一次。
   */
  async explainAppearance(id: string, candidate: unknown) {
    const candidateId = uuid(candidate as string);
    const anchorId = (await this.swipeLogs.served(id, candidateId))?.anchorId;
    if (!anchorId) return fail(404, "NOT_FOUND", "找不到這張卡的外貌比對。");
    const avatar = { where: { deletedAt: null, isAvatar: true }, take: 1 };
    const users = await this.db.user.findMany({
      where: {
        OR: [
          {
            id: candidateId,
            blocks: { none: { blockedUserId: id } },
            blockedBy: { none: { userId: id } },
          },
          { id: anchorId, ...visibleLiked(id) },
        ],
      },
      include: { photos: avatar },
    });
    const photo = (userId: string) =>
      users.find((u) => u.id === userId)?.photos[0];
    if (users.length < 2)
      return fail(404, "NOT_FOUND", "找不到這張卡的外貌比對。");
    const candidatePhoto = photo(candidateId);
    const anchorPhoto = photo(anchorId);
    if (!candidatePhoto || !anchorPhoto)
      return fail(409, "NO_AVATAR", "有一方沒有主照片，無法比對。");
    const state = await this.rec.explainAppearance(candidatePhoto, anchorPhoto);
    if (state.status === "offline")
      return fail(
        503,
        "REC_WORKER_OFFLINE",
        "外貌分析服務目前沒有在跑，無法標示相似部位。",
      );
    return {
      ...state,
      anchorUserId: anchorId,
      candidatePhotoUrl: photoView(candidatePhoto, id).url,
      anchorPhotoUrl: photoView(anchorPhoto, id).url,
    };
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
          eligible: eligible(me, u) && eligible(u, me),
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
    // 喜歡／略過真的寫進資料庫才記滑卡紀錄（已配對時一般模式不會寫）。
    let recorded = false;
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
      if (!test && (!me || !eligible(me, other) || !eligible(other, me)))
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
      recorded = true;
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
    if (recorded)
      await this.swipeLogs.record(id, dto.targetUserId, dto.action, test);
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
