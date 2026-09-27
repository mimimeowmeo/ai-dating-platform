// 滑卡紀錄：探索頁每次回傳清單時，把每張卡當下的推薦狀態（分數、推薦卡還是未推薦卡、第幾張）存在 Redis；
// 按喜歡／略過時取出那張卡的狀態，寫進 swipe_logs。用來評估推薦準不準，之後訓練個人化比重。
//
// 只保留最後一次回傳的清單：前端一次只顯示第一張，而且先送出喜歡／略過、成功後才重抓清單，
// 所以滑的一定是上一次清單裡的卡。記錄失敗只寫固定字串，不影響探索頁或滑卡。
import { Injectable } from "@nestjs/common";
import { Database, Infrastructure } from "./core";

export type ServedCard = {
  // recommended 推薦卡、exploration 未推薦卡、latest AI 排序失敗時退回最新註冊的順序
  source: "recommended" | "exploration" | "latest";
  position: number;
  rank: number;
  poolSize: number;
  score: number | null;
  appearanceWeight: number | null;
  appearancePercentile: number | null;
  appearanceSimilarity: number | null;
  passPenalty: number | null;
  nearLike: number | null;
  nearPass: number | null;
  // 最像的那位喜歡過的人；只給測試畫面用，不寫進 swipe_logs。
  anchorId: string | null;
  interestPercentile: number | null;
  interestScore: number | null;
  likeCount: number | null;
  swipeCount: number | null;
  hardFilter: boolean;
  appearanceOn: boolean;
  interestOn: boolean;
  rankingVersion: string;
  servedAt: string;
};

/** 回傳清單後多久內滑卡還對得到當時的狀態；超過就記成 unknown。 */
const SERVED_TTL_SECONDS = 24 * 60 * 60;
const servedKey = (userId: string) => `discovery:served:${userId}`;

@Injectable()
export class SwipeLogs {
  constructor(
    private db: Database,
    private infra: Infrastructure,
  ) {}

  /** 記下這次回傳的每張卡當下的狀態，取代上一次的清單。 */
  async remember(
    userId: string,
    cards: { userId: string; served: ServedCard }[],
  ) {
    if (!cards.length) return;
    const key = servedKey(userId);
    try {
      await this.infra.redis
        .multi()
        .del(key)
        .hset(
          key,
          Object.fromEntries(
            cards.map((card) => [card.userId, JSON.stringify(card.served)]),
          ),
        )
        .expire(key, SERVED_TTL_SECONDS)
        .exec();
    } catch {
      console.error("discovery_served_cache_failed");
    }
  }

  /** 上一次回傳的清單裡，這張卡當時的狀態；不在清單裡回傳 null。 */
  async served(
    userId: string,
    targetUserId: string,
  ): Promise<ServedCard | null> {
    const raw = await this.infra.redis.hget(servedKey(userId), targetUserId);
    return raw ? JSON.parse(raw) : null;
  }

  /** 喜歡／略過寫進資料庫之後呼叫。search 是從測試用搜尋列滑的，沒有排序狀態。 */
  async record(
    userId: string,
    targetUserId: string,
    action: "like" | "pass",
    search: boolean,
  ) {
    try {
      const served = search ? null : await this.served(userId, targetUserId);
      await this.db.swipeLog.create({
        data: {
          userId,
          targetUserId,
          action,
          source: search ? "search" : (served?.source ?? "unknown"),
          ...(served && {
            position: served.position,
            rank: served.rank,
            poolSize: served.poolSize,
            score: served.score,
            appearanceWeight: served.appearanceWeight,
            appearancePercentile: served.appearancePercentile,
            appearanceSimilarity: served.appearanceSimilarity,
            passPenalty: served.passPenalty,
            interestPercentile: served.interestPercentile,
            interestScore: served.interestScore,
            likeCount: served.likeCount,
            swipeCount: served.swipeCount,
            hardFilter: served.hardFilter,
            appearanceOn: served.appearanceOn,
            interestOn: served.interestOn,
            rankingVersion: served.rankingVersion,
            servedAt: new Date(served.servedAt),
          }),
        },
      });
    } catch {
      console.error("swipe_log_failed");
    }
  }
}
