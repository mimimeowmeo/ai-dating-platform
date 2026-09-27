// 把 ml/experiments/appearance_face_clip.py --export-vectors 的輸出寫進資料庫：
// - appearance_vectors_*.jsonl → appearance_embeddings（每張照片一個向量）
// - glasses_direction_*.json（選填）→ appearance_directions（worker 算的新向量寫回時要扣掉的方向）
// 用法：pnpm db:import-appearance <vectors.jsonl> [direction.json]
// 原生 SQL 照 ADR 0002 集中在 VectorStore，這裡只負責讀檔。
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Database } from "../src/core";
import { VectorStore } from "../src/ai-store";

const read = (file: string) =>
  readFileSync(resolve(process.env.INIT_CWD ?? process.cwd(), file), "utf8");

async function main() {
  const [vectorsFile, directionFile] = process.argv.slice(2);
  if (!vectorsFile)
    throw new Error("請提供 appearance_vectors_*.jsonl 的路徑。");
  const rows = read(vectorsFile)
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const row = JSON.parse(line);
      return {
        storageKey: row.storage_key,
        modelVersion: row.model_version,
        embedding: row.embedding,
      };
    });
  const db = new Database();
  const vectors = new VectorStore(db);
  try {
    const written = await vectors.upsertAppearanceEmbeddings(rows);
    console.log(
      `已匯入 ${written}／${rows.length} 筆外貌向量（資料庫裡找不到照片的略過）。`,
    );
    if (directionFile) {
      const direction = JSON.parse(read(directionFile));
      await vectors.upsertAppearanceDirection(
        direction.name,
        direction.model_version,
        direction.direction,
      );
      console.log(
        `已匯入外貌方向 ${direction.name}（${direction.model_version}）。`,
      );
    }
  } finally {
    await db.$disconnect();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
