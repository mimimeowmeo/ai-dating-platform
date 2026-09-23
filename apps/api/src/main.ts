import "reflect-metadata";
import { Module } from "@nestjs/common";
import { NestFactory } from "@nestjs/core";
import type { NestExpressApplication } from "@nestjs/platform-express";
import helmet from "helmet";
import cookieParser from "cookie-parser";
import { json } from "express";
import {
  allowedWebOrigins,
  Database,
  Infrastructure,
  Identity,
  AuthGuard,
  ErrorFilter,
  config,
} from "./core";
import { AuthController, AuthService } from "./auth";
import { HealthController, ProductController } from "./controllers";
import { Profiles } from "./profiles";
import { Social } from "./social";
import { Realtime } from "./realtime";
import { AiClient } from "./ai-client";
import { AiData } from "./ai-data";
import { AiJobs } from "./ai-jobs";
import { MessageOrigins } from "./ai-origins";
import { ReplySuggestions } from "./ai-reply";
import { VectorStore } from "./ai-store";
@Module({
  controllers: [AuthController, HealthController, ProductController],
  providers: [
    Database,
    Infrastructure,
    Identity,
    AuthGuard,
    AuthService,
    Profiles,
    Social,
    Realtime,
    // AI 推薦回覆（docs/ai/REPLY-SUGGESTIONS-SPEC.md）：
    // VectorStore 管原生 SQL 與 pgvector、AiData 組請求、AiClient 呼叫 AI 服務、
    // AiJobs 負責背景工作與結果寫回、MessageOrigins 標記訊息來源。
    VectorStore,
    AiData,
    AiClient,
    AiJobs,
    MessageOrigins,
    ReplySuggestions,
  ],
})
class AppModule {}
async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    bodyParser: false,
    logger: ["log", "warn", "error"],
  });
  // 前提是請求只經過一層代理；若把 API 埠直接公開，使用者就能偽造 X-Forwarded-For。
  // nginx 前面還有 cloudflared 時，由 nginx 還原來源位址（infrastructure/nginx/client-ip/）。
  app.set("trust proxy", 1);
  app.setGlobalPrefix("api/v1");
  app.use(helmet({ crossOriginResourcePolicy: { policy: "same-origin" } }));
  app.use(json({ limit: "32kb" }));
  app.use(cookieParser());
  app.enableCors({ origin: allowedWebOrigins, credentials: true });
  app.useGlobalFilters(new ErrorFilter());
  app.enableShutdownHooks();
  await app.listen(config.PORT, "0.0.0.0");
}
bootstrap().catch(() => {
  console.error("啟動失敗：請確認資料庫、儲存服務與環境設定。");
  process.exit(1);
});
