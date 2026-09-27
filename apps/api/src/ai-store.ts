// pgvector 與其他原生 SQL 的唯一出入口。
//
// ADR 0002：資料庫只由 NestJS 存取，而且所有原生 SQL 集中在這個 service，不散落各處。
// 會用原生 SQL 的原因有兩個：
// 1. 向量欄位在 Prisma 是 Unsupported("vector(768)")，Prisma Client 讀不到也寫不了。
// 2. 相似度排序（<=>）與「取每位使用者最新版本」這類查詢，用 SQL 比在程式裡處理快很多。
import { Injectable } from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { Database } from "./core";
import { toVector } from "./ai-text";
import type {
  AiChunkResult,
  AiStyleCard,
  AiStyleFacet,
  AiStyleProfileResult,
} from "./ai-client";

/** 外貌向量的維度（CLIP ViT-B/32 的影像向量）；必須和 appearance_embeddings 的 vector(512) 一致。 */
export const APPEARANCE_DIMENSIONS = 512;
/** 產生推薦時最多放幾段檢索到的舊對話（AI 服務的上限是 12 段，prompt 預算約 1.5k tokens）。 */
export const CHUNK_MATCHES = 6;
/** 產生推薦時最多放幾條 B 的特徵句（AI 服務的上限是 20 條）。 */
export const FACET_MATCHES = 8;

/** 風格卡在資料庫裡的樣子（user_style_profiles 一列）。 */
type ProfileRow = {
  version: number;
  feature_version: string;
  confidence: "high" | "low" | "none";
  sample_source: "chat" | "mixed" | "bio" | "none";
  message_count: number;
  window_from: Date | null;
  window_to: Date | null;
  stats: AiStyleCard["stats"];
  voice_notes: string[];
  bio_sample: string;
  model_name: string | null;
  prompt_version: string | null;
  created_at: Date;
};

@Injectable()
export class VectorStore {
  constructor(private db: Database) {}

  /**
   * 把字串陣列變成 SQL 的 text[]，每個元素都是參數（不會有跳脫字元的問題）。
   * 空陣列要特別處理：`ARRAY[]` 沒有型別，Postgres 會拒絕。
   */
  private textArray(values: string[]) {
    return values.length
      ? Prisma.sql`ARRAY[${Prisma.join(values)}]::text[]`
      : Prisma.sql`ARRAY[]::text[]`;
  }

  /**
   * 用查詢向量找出這個聊天室最相關的幾段舊對話（RAG 的檢索步驟）。
   *
   * `<=>` 是 pgvector 的 cosine 距離（0 最相似），1 - 距離就是相似度，直接當分數傳給 AI 服務。
   * 只搜同一個聊天室、而且有向量的片段；每個聊天室只有幾十到幾千段，先用精確搜尋就夠快，
   * 等資料量大到變慢再加 HNSW 索引（ADR 0002）。
   */
  async searchChunks(conversationId: string, query: number[], limit = CHUNK_MATCHES) {
    const rows = await this.db.$queryRaw<
      { content: string; last_at: Date; score: number }[]
    >`
      SELECT content, last_at, 1 - (embedding <=> ${toVector(query)}::vector) AS score
      FROM conversation_chunks
      WHERE conversation_id = ${conversationId}::uuid AND embedding IS NOT NULL
      ORDER BY embedding <=> ${toVector(query)}::vector
      LIMIT ${limit}
    `;
    return rows.map((row) => ({
      content: row.content,
      lastAt: row.last_at.toISOString(),
      score: Number(row.score),
    }));
  }

  /**
   * 找出某位使用者風格卡上、跟目前話題最相關的幾條特徵句。
   *
   * 只看指定版本（最新版），避免舊版本的特徵句混進來。沒有向量的特徵句（向量化失敗時）
   * 會被排除；那種情況下 prompt 仍然有風格卡的數值與語氣，只是少了話題相關的細節。
   */
  async searchFacets(
    userId: string,
    profileVersion: number,
    query: number[],
    limit = FACET_MATCHES,
  ): Promise<AiStyleFacet[]> {
    const rows = await this.db.$queryRaw<
      { kind: AiStyleFacet["kind"]; statement: string; weight: number; evidence: number }[]
    >`
      SELECT kind, statement, weight, evidence
      FROM user_style_facets
      WHERE user_id = ${userId}::uuid
        AND profile_version = ${profileVersion}
        AND embedding IS NOT NULL
      ORDER BY embedding <=> ${toVector(query)}::vector
      LIMIT ${limit}
    `;
    return rows.map((row) => ({
      kind: row.kind,
      statement: row.statement,
      weight: Number(row.weight),
      evidence: Number(row.evidence),
    }));
  }

  /**
   * 讀出使用者最新版的風格卡（不含向量），組成 AI 服務要的 StyleCard 格式。
   *
   * facets 全部帶上（一張卡最多 36 條，AI 服務的上限是 40），順序是越近越優先：
   * 依最後出現時間由新到舊，一樣新再依權重（2026-09-27）。回覆模式由 AI 服務自己依權重挑，
   * 跟話題相關的部分由 searchFacets 另外補；開場、追問、重啟則直接依這個順序找話題。
   * 沒有風格卡時回傳 null，AI 服務會改用 bio 做冷啟動（規格 5.3）。
   * createdAt 是這張卡建立的時間，用來判斷要不要重新萃取（見 AiJobs.enqueueStyleIfStale）。
   */
  async styleCard(userId: string, facetLimit = 40) {
    const [profile] = await this.db.$queryRaw<ProfileRow[]>`
      SELECT version, feature_version, confidence, sample_source, message_count,
             window_from, window_to, stats, voice_notes, bio_sample, model_name, prompt_version,
             created_at
      FROM user_style_profiles
      WHERE user_id = ${userId}::uuid
      ORDER BY version DESC
      LIMIT 1
    `;
    if (!profile) return null;
    const facets = await this.db.$queryRaw<
      {
        kind: AiStyleFacet["kind"];
        statement: string;
        weight: number;
        evidence: number;
        last_seen_at: Date | null;
      }[]
    >`
      SELECT kind, statement, weight, evidence, last_seen_at
      FROM user_style_facets
      WHERE user_id = ${userId}::uuid AND profile_version = ${profile.version}
      ORDER BY last_seen_at DESC NULLS LAST, weight DESC, statement ASC
      LIMIT ${facetLimit}
    `;
    const card: AiStyleCard = {
      featureVersion: profile.feature_version,
      userId,
      confidence: profile.confidence,
      sampleSource: profile.sample_source,
      stats: profile.stats,
      voiceNotes: profile.voice_notes ?? [],
      facets: facets.map((facet) => ({
        kind: facet.kind,
        statement: facet.statement,
        weight: Number(facet.weight),
        evidence: Number(facet.evidence),
        // 沒有時間（舊版卡片、只來自自我介紹）就不送這個欄位：AI 服務拒絕不認得的欄位，
        // 這樣後端先更新、AI 服務還是舊版時，舊的風格卡照樣能用。
        ...(facet.last_seen_at
          ? { lastSeenAt: facet.last_seen_at.toISOString() }
          : {}),
      })),
      bioSample: profile.bio_sample,
      messageCount: profile.message_count,
      windowFrom: profile.window_from?.toISOString() ?? null,
      windowTo: profile.window_to?.toISOString() ?? null,
      modelName: profile.model_name,
      promptVersion: profile.prompt_version,
    };
    return { version: profile.version, card, createdAt: profile.created_at };
  }

  /**
   * 寫入一批新的對話切片（chunk-embed 的結果）。
   *
   * 切片是增量重算的：後端從「上一段還沒結束的片段」開始把訊息送給 AI 服務，
   * 所以要先刪掉重疊的舊片段再寫入，否則同一段對話會有兩份。
   * 刪除條件是「還沒結束的片段」或「開始時間在這批之後的片段」，兩者都會被這批取代。
   * 整批放在同一個交易裡：中途失敗時不會只刪不寫。
   */
  async replaceChunks(result: AiChunkResult) {
    if (!result.chunks.length) return 0;
    const since = new Date(result.chunks[0].firstAt);
    return this.db.$transaction(async (tx) => {
      await tx.$executeRaw`
        DELETE FROM conversation_chunks
        WHERE conversation_id = ${result.conversationId}::uuid
          AND (is_open = true OR first_at >= ${since})
      `;
      for (const chunk of result.chunks) {
        // 沒有向量時也要存：內容本身仍可用來顯示或日後補算，只是這次檢索不到。
        const embedding = chunk.vector
          ? Prisma.sql`${toVector(chunk.vector)}::vector`
          : Prisma.sql`NULL`;
        await tx.$executeRaw`
          INSERT INTO conversation_chunks (
            id, conversation_id, first_message_id, last_message_id, first_at, last_at,
            message_count, content, token_estimate, is_open, chunk_version, embedding_model, embedding
          ) VALUES (
            gen_random_uuid(), ${result.conversationId}::uuid,
            ${chunk.firstMessageId}::uuid, ${chunk.lastMessageId}::uuid,
            ${new Date(chunk.firstAt)}, ${new Date(chunk.lastAt)},
            ${chunk.messageCount}, ${chunk.content}, ${chunk.tokenEstimate},
            ${chunk.isOpen}, ${result.chunkVersion}, ${result.embeddingModel ?? null}, ${embedding}
          )
        `;
      }
      return result.chunks.length;
    });
  }

  /**
   * 找出「還沒結束的片段」從哪一則訊息開始，下次切片就從那裡重算（規格 5.2）。
   * 沒有任何片段時回傳 null，代表整個聊天室都要重新切。
   */
  async openChunkStart(conversationId: string) {
    const [row] = await this.db.$queryRaw<{ first_message_id: string; first_at: Date }[]>`
      SELECT first_message_id, first_at
      FROM conversation_chunks
      WHERE conversation_id = ${conversationId}::uuid AND is_open = true
      ORDER BY first_at ASC
      LIMIT 1
    `;
    return row ? { messageId: row.first_message_id, at: row.first_at } : null;
  }

  /**
   * 寫入一張新的風格卡（build-style 的結果），回傳新版本號。
   *
   * 版本號用 `MAX(version) + 1` 在同一句 INSERT 裡算出來，不會有兩個工作同時寫入撞號的問題。
   * 特徵句與它的向量一一對應（AI 服務保證順序相同），向量化失敗時 facetVectors 是 null，
   * 這時仍然寫入特徵句，只是 embedding 留空、檢索不到。
   */
  async insertStyleProfile(userId: string, result: AiStyleProfileResult) {
    const card = result.card;
    return this.db.$transaction(async (tx) => {
      const [created] = await tx.$queryRaw<{ version: number }[]>`
        INSERT INTO user_style_profiles (
          user_id, version, feature_version, confidence, sample_source, message_count,
          window_from, window_to, stats, voice_notes, bio_sample, model_name, prompt_version, embedding_model
        )
        SELECT ${userId}::uuid, COALESCE(MAX(version), 0) + 1, ${card.featureVersion},
               ${card.confidence}, ${card.sampleSource}, ${card.messageCount},
               ${card.windowFrom ? new Date(card.windowFrom) : null},
               ${card.windowTo ? new Date(card.windowTo) : null},
               ${JSON.stringify(card.stats)}::jsonb, ${this.textArray(card.voiceNotes)},
               ${card.bioSample}, ${card.modelName ?? null}, ${card.promptVersion ?? null},
               ${result.embeddingModel ?? null}
        FROM user_style_profiles WHERE user_id = ${userId}::uuid
        RETURNING version
      `;
      const version = created.version;
      for (const [index, facet] of card.facets.entries()) {
        const vector = result.facetVectors?.[index];
        const embedding = vector
          ? Prisma.sql`${toVector(vector)}::vector`
          : Prisma.sql`NULL`;
        await tx.$executeRaw`
          INSERT INTO user_style_facets (
            id, user_id, profile_version, kind, statement, weight, evidence, last_seen_at, embedding
          ) VALUES (
            gen_random_uuid(), ${userId}::uuid, ${version}, ${facet.kind},
            ${facet.statement}, ${facet.weight}, ${facet.evidence},
            ${facet.lastSeenAt ? new Date(facet.lastSeenAt) : null}, ${embedding}
          )
        `;
      }
      return version;
    });
  }

  /**
   * 找出「使用者自己發起的 AI 話題區段」涵蓋到哪些訊息（規格 5.3）。
   *
   * 區段是用首末訊息記錄的，所以這裡用首末訊息的時間當範圍，挑出同一個聊天室裡
   * 發起者自己在那段期間發的訊息。只標記發起者本人：對方的回覆不算（使用者的決定）。
   * 回傳訊息 id 集合，萃取風格卡時用來把這些訊息標成 inAiTopic（話題喜好降權 ×0.3）。
   */
  async aiTopicMessageIds(userId: string, since: Date | null) {
    const rows = await this.db.$queryRaw<{ id: string }[]>`
      SELECT m.id
      FROM ai_topic_spans s
      JOIN messages start_message ON start_message.id = s.start_message_id
      JOIN messages end_message ON end_message.id = s.end_message_id
      JOIN messages m ON m.conversation_id = s.conversation_id
                     AND m.sender_id = s.initiator_id
                     AND m.created_at BETWEEN start_message.created_at AND end_message.created_at
      WHERE s.initiator_id = ${userId}::uuid
        AND (${since}::timestamp IS NULL OR m.created_at >= ${since}::timestamp)
    `;
    return new Set(rows.map((row) => row.id));
  }

  /**
   * 全站的聊天寫法平均值，冷啟動時當數值目標（規格 5.3）。
   *
   * 只取最近 5,000 則真人訊息，避免整表掃描；emoji 與連發則數在 SQL 裡不好算，
   * 留給 AI 服務的內建預設值（那組數字本來就是從這個資料庫量出來的）。
   * 呼叫端會把結果快取起來，這個查詢不需要每次請求都跑。
   */
  async siteStats() {
    const [row] = await this.db.$queryRaw<
      {
        message_count: bigint;
        median_chars: number | null;
        mean_chars: number | null;
        question_ratio: number | null;
        exclamation_ratio: number | null;
        laughter_ratio: number | null;
      }[]
    >`
      WITH recent AS (
        SELECT m.content
        FROM messages m
        LEFT JOIN message_origins o ON o.message_id = m.id
        WHERE COALESCE(o.origin, 'human') = 'human'
        ORDER BY m.created_at DESC
        LIMIT 5000
      )
      SELECT count(*) AS message_count,
             percentile_cont(0.5) WITHIN GROUP (ORDER BY length(content)) AS median_chars,
             avg(length(content)) AS mean_chars,
             avg(CASE WHEN content ~ '[?？]' THEN 1 ELSE 0 END) AS question_ratio,
             avg(CASE WHEN content ~ '[!！]' THEN 1 ELSE 0 END) AS exclamation_ratio,
             avg(CASE WHEN content ~ '哈哈|呵呵|嘻嘻|笑死|ㄏㄏ' THEN 1 ELSE 0 END) AS laughter_ratio
      FROM recent
    `;
    const count = Number(row?.message_count ?? 0);
    if (!count) return null;
    return {
      messageCount: count,
      medianChars: Number(row.median_chars ?? 0),
      meanChars: Number(row.mean_chars ?? 0),
      questionRatio: Number(row.question_ratio ?? 0),
      exclamationRatio: Number(row.exclamation_ratio ?? 0),
      laughterRatio: Number(row.laughter_ratio ?? 0),
    };
  }

  /**
   * 外貌分數與略過扣分要用的數字，每位有向量的候選人一列：
   * - base：和「我最近按喜歡的人」的主照片向量逐一比，取最高的相似度（多錨點、不取平均：
   *   平均會把喜歡的兩種型混成四不像，HeartLink Appearance Core V1）。
   * - nearLike／nearPass：相似度達到「附近」門檻的喜歡、略過各幾個，給 PASS V2 扣分用。
   *   門檻依候選人性別不同（near），由呼叫端依種子資料校準後傳入；gender 給扣分換算尺度用。
   * 沒有向量的候選人不會出現在結果裡；候選池最多一萬多人，精確搜尋就夠快。
   */
  async appearanceScores(
    anchorUserIds: string[],
    passUserIds: string[],
    candidateIds: string[],
    near: { woman: number; man: number; other: number },
  ) {
    const scores = new Map<
      string,
      { base: number; nearLike: number; nearPass: number; gender: string }
    >();
    if (!anchorUserIds.length || !candidateIds.length) return scores;
    const rows = await this.db.$queryRaw<
      {
        user_id: string;
        gender: string;
        base: number | null;
        near_like: bigint;
        near_pass: bigint;
      }[]
    >`
      WITH anchors AS (
        SELECT e.embedding
        FROM appearance_embeddings e
        JOIN user_photos p ON p.id = e.photo_id AND p.is_avatar AND p.deleted_at IS NULL
        WHERE e.user_id = ANY(${anchorUserIds}::uuid[]) AND e.embedding IS NOT NULL
      ),
      passes AS (
        SELECT e.embedding
        FROM appearance_embeddings e
        JOIN user_photos p ON p.id = e.photo_id AND p.is_avatar AND p.deleted_at IS NULL
        WHERE e.user_id = ANY(${passUserIds}::uuid[]) AND e.embedding IS NOT NULL
      ),
      candidates AS (
        SELECT e.user_id, e.embedding, pr.gender,
               CASE pr.gender WHEN 'woman' THEN ${near.woman}::float8
                              WHEN 'man' THEN ${near.man}::float8
                              ELSE ${near.other}::float8 END AS near
        FROM appearance_embeddings e
        JOIN user_photos p ON p.id = e.photo_id AND p.is_avatar AND p.deleted_at IS NULL
        JOIN profiles pr ON pr.user_id = e.user_id
        WHERE e.user_id = ANY(${candidateIds}::uuid[]) AND e.embedding IS NOT NULL
      )
      SELECT c.user_id::text AS user_id, c.gender::text AS gender,
             (SELECT MAX(1 - (c.embedding <=> a.embedding)) FROM anchors a) AS base,
             (SELECT COUNT(*) FROM anchors a WHERE 1 - (c.embedding <=> a.embedding) >= c.near) AS near_like,
             (SELECT COUNT(*) FROM passes s WHERE 1 - (c.embedding <=> s.embedding) >= c.near) AS near_pass
      FROM candidates c
    `;
    for (const row of rows)
      if (row.base !== null)
        scores.set(row.user_id, {
          base: Number(row.base),
          nearLike: Number(row.near_like),
          nearPass: Number(row.near_pass),
          gender: row.gender,
        });
    return scores;
  }

  /**
   * 興趣分數：五類標籤各算 Jaccard（共同標籤 ÷ 兩人標籤聯集），再依 70/15/2/5/8 加權
   * （HeartLink Matching Core 的 PREFERENCE_WEIGHTS）。dating_goal 是硬篩選，不計分。
   * 沒有任何計分標籤的候選人不會出現在結果裡，呼叫端當 0 分。
   */
  async interestScores(userId: string, candidateIds: string[]) {
    if (!candidateIds.length) return new Map<string, number>();
    const rows = await this.db.$queryRaw<{ user_id: string; score: number }[]>`
      WITH weights(category, weight) AS (
        VALUES ('interest', 0.70), ('personality', 0.15), ('lifestyle', 0.02),
               ('value', 0.05), ('diet', 0.08)
      ),
      mine AS (
        SELECT t.category, ut.trait_id
        FROM user_traits ut JOIN traits t ON t.id = ut.trait_id
        WHERE ut.user_id = ${userId}::uuid
      ),
      mine_count AS (SELECT category, COUNT(*) AS n FROM mine GROUP BY category),
      theirs AS (
        SELECT ut.user_id, t.category, COUNT(*) AS n, COUNT(m.trait_id) AS shared
        FROM user_traits ut
        JOIN traits t ON t.id = ut.trait_id
        LEFT JOIN mine m ON m.trait_id = ut.trait_id
        WHERE ut.user_id = ANY(${candidateIds}::uuid[])
        GROUP BY ut.user_id, t.category
      )
      SELECT th.user_id::text AS user_id,
             SUM(w.weight * th.shared::float8 / (th.n + COALESCE(mc.n, 0) - th.shared)) AS score
      FROM theirs th
      JOIN weights w ON w.category = th.category
      LEFT JOIN mine_count mc ON mc.category = th.category
      GROUP BY th.user_id
    `;
    return new Map(rows.map((row) => [row.user_id, Number(row.score)]));
  }

  /**
   * 匯入外貌向量（prisma/import-appearance.ts 用）：用照片的 storage_key 對到 user_photos，
   * 對不到的照片略過；同一張照片重匯就覆蓋。回傳實際寫入的筆數。
   */
  async upsertAppearanceEmbeddings(
    rows: { storageKey: string; modelVersion: string; embedding: number[] }[],
  ) {
    let written = 0;
    for (const row of rows) {
      written += await this.db.$executeRaw`
        INSERT INTO appearance_embeddings (photo_id, user_id, model_version, embedding)
        SELECT p.id, p.user_id, ${row.modelVersion},
               ${toVector(row.embedding, APPEARANCE_DIMENSIONS)}::vector
        FROM user_photos p
        WHERE p.storage_key = ${row.storageKey}
        ON CONFLICT (photo_id) DO UPDATE
          SET model_version = EXCLUDED.model_version, embedding = EXCLUDED.embedding,
              created_at = CURRENT_TIMESTAMP
      `;
    }
    return written;
  }

  /**
   * worker 算好的向量寫回（rec-jobs.ts）；找不到臉時 embedding 是 null，也留一列表示處理過。
   * 照片已刪除或不存在就不寫，避免刪照片後向量又被寫回來；FOR SHARE 等正在刪這張照片的交易結束再判斷，
   * 不會在刪除交易的空檔寫進去。回傳是否寫入。
   */
  async storeAppearanceEmbedding(
    photoId: string,
    modelVersion: string,
    embedding: number[] | null,
  ) {
    const vector = embedding
      ? toVector(embedding, APPEARANCE_DIMENSIONS)
      : null;
    const written = await this.db.$executeRaw`
      INSERT INTO appearance_embeddings (photo_id, user_id, model_version, embedding)
      SELECT p.id, p.user_id, ${modelVersion}, ${vector}::vector
      FROM user_photos p
      WHERE p.id = ${photoId}::uuid AND p.deleted_at IS NULL
      FOR SHARE
      ON CONFLICT (photo_id) DO UPDATE
        SET model_version = EXCLUDED.model_version, embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
    `;
    return written > 0;
  }

  /** 讀某個外貌方向（例如 glasses）；沒有匯入過就回 null。 */
  async appearanceDirection(name: string) {
    const [row] = await this.db.$queryRaw<
      { model_version: string; direction: string }[]
    >`
      SELECT model_version, direction::text AS direction
      FROM appearance_directions
      WHERE name = ${name}
    `;
    return row
      ? {
          modelVersion: row.model_version,
          vector: JSON.parse(row.direction) as number[],
        }
      : null;
  }

  /** 匯入外貌方向（prisma/import-appearance.ts 用）；同名就覆蓋。 */
  async upsertAppearanceDirection(
    name: string,
    modelVersion: string,
    vector: number[],
  ) {
    await this.db.$executeRaw`
      INSERT INTO appearance_directions (name, model_version, direction)
      VALUES (${name}, ${modelVersion}, ${toVector(vector, APPEARANCE_DIMENSIONS)}::vector)
      ON CONFLICT (name) DO UPDATE
        SET model_version = EXCLUDED.model_version, direction = EXCLUDED.direction,
            created_at = CURRENT_TIMESTAMP
    `;
  }
}
