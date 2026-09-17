import { existsSync, readFileSync, writeFileSync, chmodSync } from "node:fs";
import { randomBytes } from "node:crypto";
import { parseEnv } from "node:util";

const envPath = new URL("../.env", import.meta.url);
let content = existsSync(envPath) ? readFileSync(envPath, "utf8") : "";
const env = parseEnv(content);
const secret = () => randomBytes(32).toString("hex");
const additions = [];
function ensure(key, value) {
  if (Object.hasOwn(env, key) && env[key] !== "") return;
  if (Object.hasOwn(env, key)) {
    content = content.replace(
      new RegExp(`^${key}=.*$`, "m"),
      `${key}=${value}`,
    );
  } else additions.push(`${key}=${value}`);
  env[key] = value;
}
ensure("POSTGRES_DB", "dating");
ensure("POSTGRES_USER", "dating");
ensure("POSTGRES_PASSWORD", secret());
ensure("POSTGRES_PORT", "5432");
const credentials = `${encodeURIComponent(env.POSTGRES_USER)}:${encodeURIComponent(env.POSTGRES_PASSWORD)}`;
const database = encodeURIComponent(env.POSTGRES_DB);
ensure(
  "DATABASE_URL",
  `postgresql://${credentials}@127.0.0.1:${env.POSTGRES_PORT}/${database}?schema=public`,
);
ensure(
  "DOCKER_DATABASE_URL",
  `postgresql://${credentials}@postgres:5432/${database}?schema=public`,
);
ensure("REDIS_URL", "redis://127.0.0.1:6379");
ensure("S3_ENDPOINT", "http://127.0.0.1:9000");
ensure("S3_ACCESS_KEY", "dating-local");
ensure("S3_SECRET_KEY", secret());
ensure("S3_BUCKET", "dating-media");
ensure("JWT_SECRET", secret());
ensure("AI_INTERNAL_URL", "http://127.0.0.1:8000");
ensure("AI_INTERNAL_TOKEN", secret());
ensure("WEB_ORIGIN", "http://localhost:8080");
ensure("APP_PORT", "8080");
ensure("PORT", "3001");
ensure("NODE_ENV", "development");
writeFileSync(
  envPath,
  `${content.trimEnd()}\n${additions.length ? `\n# Phase 1–7 本機開發設定\n${additions.join("\n")}\n` : ""}`,
  { mode: 0o600 },
);
chmodSync(envPath, 0o600);
console.log(
  `本機 .env 已準備完成，新增 ${additions.length} 個設定；既有非空設定保留，未輸出機密。`,
);
