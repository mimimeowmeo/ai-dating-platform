import { webOrigins } from "./web-origins";
import {
  Injectable,
  OnModuleInit,
  OnModuleDestroy,
  HttpException,
  CanActivate,
  ExecutionContext,
  Catch,
  ExceptionFilter,
  ArgumentsHost,
} from "@nestjs/common";
import { PrismaClient, Prisma } from "@prisma/client";
import Redis from "ioredis";
import { Client } from "minio";
import { z } from "zod";
import { verify } from "jsonwebtoken";
import type { Request } from "express";

const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  REDIS_URL: z.string().url(),
  S3_ENDPOINT: z.string().url(),
  S3_ACCESS_KEY: z.string().min(3),
  S3_SECRET_KEY: z.string().min(8),
  S3_BUCKET: z.string().default("dating-media"),
  JWT_SECRET: z.string().min(32),
  AI_INTERNAL_URL: z.string().url(),
  AI_INTERNAL_TOKEN: z.string().min(24),
  WEB_ORIGIN: z.string().url(),
  PORT: z.coerce.number().default(3001),
  COOKIE_SECURE: z.string().default("false"),
});
export const config = envSchema.parse(process.env);
export const allowedWebOrigins = webOrigins(config.WEB_ORIGIN);
export const fail = (status: number, code: string, message: string): never => {
  throw new HttpException({ code, message }, status);
};
const fieldLabels: Record<string, string> = {
  email: "電子郵件",
  password: "密碼",
  displayName: "顯示名稱",
  birthDate: "生日",
  gender: "性別",
  bio: "自我介紹",
  city: "居住城市",
  latitude: "緯度",
  longitude: "經度",
  datingIntent: "想遇見的關係",
  heightCm: "身高",
  traits: "我的小熱愛",
  datingGoals: "想遇見的關係",
  occupation: "職業",
  education: "學歷",
  interests: "興趣",
  hobbies: "休閒活動",
  foods: "喜愛的食物",
  values: "標籤",
  minAge: "最小年齡",
  maxAge: "最大年齡",
  preferredGender: "性別偏好",
  maxDistanceKm: "探索距離",
  minHeightCm: "最低身高",
  maxHeightCm: "最高身高",
  preferredDatingIntent: "關係期待",
  targetUserId: "對象",
  action: "操作",
  blockedUserId: "封鎖對象",
  matchId: "配對",
  conversationId: "聊天室",
  content: "訊息內容",
  clientId: "訊息識別碼",
  isTyping: "輸入狀態",
};
function issueText(issue: z.core.$ZodIssue) {
  const key = issue.path
    .filter((p): p is string => typeof p === "string")
    .at(-1);
  const label = key ? (fieldLabels[key] ?? "欄位") : "";
  // 專案自訂的 schema 訊息已是中文，保留原文，只在缺少欄位名稱時補上。
  if (/\p{Script=Han}/u.test(issue.message))
    return label && !issue.message.includes(label)
      ? `${label}：${issue.message}`
      : issue.message;
  const subject = label || "資料";
  switch (issue.code) {
    case "too_small":
      return issue.origin === "string"
        ? `${subject}至少需要 ${issue.minimum} 個字元`
        : issue.origin === "array"
          ? `${subject}至少需要選擇 ${issue.minimum} 項`
          : `${subject}不得小於 ${issue.minimum}`;
    case "too_big":
      return issue.origin === "string"
        ? `${subject}最多 ${issue.maximum} 個字元`
        : issue.origin === "array"
          ? `${subject}最多選擇 ${issue.maximum} 項`
          : `${subject}不得大於 ${issue.maximum}`;
    case "invalid_type":
      return `${subject}未填寫或格式不正確`;
    case "invalid_value":
      return `${subject}的選項不正確`;
    case "unrecognized_keys":
      return "包含不允許修改的欄位";
    default:
      return `${subject}格式不正確`;
  }
}
export function parse<T>(schema: z.ZodType<T>, body: unknown): T {
  const r = schema.safeParse(body);
  if (!r.success)
    fail(
      400,
      "VALIDATION_ERROR",
      [...new Set(r.error.issues.map(issueText))].join("；"),
    );
  return r.data!;
}
export const uuid = (id: string) =>
  parse(z.string().uuid("識別碼格式不正確"), id);
export type AuthRequest = Request & { userId: string; sessionId: string };
export const catalogs = {
  interests: ["旅行", "咖啡", "電影", "音樂", "藝術", "科技", "閱讀", "戶外"],
  hobbies: ["攝影", "登山", "健身", "烹飪", "繪畫", "桌遊", "瑜珈", "散步"],
  foods: [
    "日式料理",
    "義式料理",
    "台灣小吃",
    "甜點",
    "蔬食",
    "火鍋",
    "咖哩",
    "海鮮",
  ],
};
@Injectable()
export class Database
  extends PrismaClient
  implements OnModuleInit, OnModuleDestroy
{
  async onModuleInit() {
    await this.$connect();
  }
  async onModuleDestroy() {
    await this.$disconnect();
  }
}
@Injectable()
export class Infrastructure implements OnModuleInit, OnModuleDestroy {
  redis = new Redis(config.REDIS_URL, { maxRetriesPerRequest: 2 });
  storage: Client;
  constructor() {
    const url = new URL(config.S3_ENDPOINT);
    this.storage = new Client({
      endPoint: url.hostname,
      port: Number(url.port || (url.protocol === "https:" ? 443 : 80)),
      useSSL: url.protocol === "https:",
      accessKey: config.S3_ACCESS_KEY,
      secretKey: config.S3_SECRET_KEY,
    });
    this.redis.on("error", () => {});
  }
  async onModuleInit() {
    if (!(await this.storage.bucketExists(config.S3_BUCKET)))
      await this.storage.makeBucket(config.S3_BUCKET);
  }
  async onModuleDestroy() {
    await this.redis.quit();
  }
  async limit(key: string, max = 120, seconds = 60) {
    const count = await this.redis.incr(`rate:${key}`);
    if (count === 1) await this.redis.expire(`rate:${key}`, seconds);
    if (count > max) fail(429, "RATE_LIMIT", "操作太頻繁，請稍後再試。");
  }
}
@Injectable()
export class Identity {
  constructor(private db: Database) {}
  async authenticate(token: string) {
    let claims: any;
    try {
      claims = verify(token, config.JWT_SECRET, {
        algorithms: ["HS256"],
        issuer: "dating-api",
        audience: "dating-web",
      });
    } catch {
      return fail(401, "UNAUTHORIZED", "請重新登入。");
    }
    if (typeof claims.sub !== "string" || typeof claims.sid !== "string")
      return fail(401, "UNAUTHORIZED", "登入資訊無效。");
    const session = await this.db.session.findUnique({
      where: { id: claims.sid },
    });
    if (
      !session ||
      session.userId !== claims.sub ||
      session.revokedAt ||
      session.expiresAt < new Date()
    )
      return fail(401, "UNAUTHORIZED", "登入已失效。");
    return { userId: claims.sub as string, sessionId: claims.sid as string };
  }
}
@Injectable()
export class AuthGuard implements CanActivate {
  constructor(
    private identity: Identity,
    private infra: Infrastructure,
  ) {}
  async canActivate(context: ExecutionContext) {
    const req = context.switchToHttp().getRequest<AuthRequest>();
    const token = req.headers.authorization?.replace(/^Bearer /, "");
    if (!token) return fail(401, "UNAUTHORIZED", "請先登入。");
    Object.assign(req, await this.identity.authenticate(token));
    await this.infra.limit(`user:${req.userId}`);
    return true;
  }
}
function frameworkError(status: number) {
  if (status === 413)
    return {
      code: "PAYLOAD_TOO_LARGE",
      message: "照片太大，請選擇 8 MB 以內的檔案。",
    };
  if (status === 401) return { code: "UNAUTHORIZED", message: "請先登入。" };
  if (status === 403)
    return { code: "FORBIDDEN", message: "沒有權限執行此操作。" };
  if (status === 404)
    return { code: "NOT_FOUND", message: "找不到指定的資源。" };
  if (status >= 500)
    return {
      code: "INTERNAL_ERROR",
      message: "服務暫時無法處理，請稍後再試。",
    };
  return { code: "BAD_REQUEST", message: "請求格式不正確。" };
}
function parserErrorStatus(error: unknown) {
  const e = error as { status?: unknown; type?: unknown } | null;
  return e &&
    typeof e.status === "number" &&
    typeof e.type === "string" &&
    /^(entity|request|encoding|charset|parameters|stream)\./.test(e.type)
    ? e.status
    : undefined;
}
@Catch()
export class ErrorFilter implements ExceptionFilter {
  catch(error: unknown, host: ArgumentsHost) {
    const res = host.switchToHttp().getResponse();
    if (error instanceof HttpException) {
      const body = error.getResponse();
      return res
        .status(error.getStatus())
        .json(
          typeof body === "object" && body !== null && "code" in body
            ? body
            : frameworkError(error.getStatus()),
        );
    }
    const parserStatus = parserErrorStatus(error);
    if (parserStatus === 413)
      return res
        .status(413)
        .json({ code: "PAYLOAD_TOO_LARGE", message: "送出的內容太大。" });
    if (parserStatus && parserStatus >= 400 && parserStatus < 500)
      return res.status(parserStatus).json(frameworkError(parserStatus));
    if (error instanceof Prisma.PrismaClientKnownRequestError) {
      if (error.code === "P2002")
        return res
          .status(409)
          .json({ code: "CONFLICT", message: "此資料已存在。" });
      if (error.code === "P2025")
        return res
          .status(404)
          .json({ code: "NOT_FOUND", message: "找不到指定資料。" });
    }
    console.error(
      "request_failed",
      error instanceof Error ? error.name : "UnknownError",
    );
    return res.status(500).json({
      code: "INTERNAL_ERROR",
      message: "服務暫時無法處理，請稍後再試。",
    });
  }
}
export function ageAt(birth: Date, now = new Date()) {
  let age = now.getUTCFullYear() - birth.getUTCFullYear();
  if (
    now.getUTCMonth() < birth.getUTCMonth() ||
    (now.getUTCMonth() === birth.getUTCMonth() &&
      now.getUTCDate() < birth.getUTCDate())
  )
    age--;
  return age;
}
export function distance(
  a: { latitude: number; longitude: number },
  b: { latitude: number; longitude: number },
) {
  const r = Math.PI / 180,
    dLat = (b.latitude - a.latitude) * r,
    dLon = (b.longitude - a.longitude) * r;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a.latitude * r) *
      Math.cos(b.latitude * r) *
      Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(Math.min(1, h)));
}
export async function pairLock(
  tx: Prisma.TransactionClient,
  a: string,
  b: string,
) {
  const key = [a, b].sort().join(":");
  await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${key}))`;
}
