import { PrismaClient } from "@prisma/client";
import { Client } from "minio";
const run = process.env.E2E_RUN_ID;
if (!run || !/^[0-9a-f-]{36}$/.test(run))
  throw new Error("缺少本次測試識別碼，拒絕清理。");
const db = new PrismaClient();
try {
  const where = {
    email: { startsWith: `e2e-${run}-`, endsWith: "@example.test" },
  };
  const photos = await db.photo.findMany({
    where: { user: where },
    select: { storageKey: true },
  });
  if (photos.length) {
    const url = new URL(process.env.S3_ENDPOINT);
    const storage = new Client({
      endPoint: url.hostname,
      port: Number(url.port || (url.protocol === "https:" ? 443 : 80)),
      useSSL: url.protocol === "https:",
      accessKey: process.env.S3_ACCESS_KEY,
      secretKey: process.env.S3_SECRET_KEY,
    });
    await storage.removeObjects(
      process.env.S3_BUCKET || "dating-media",
      photos.map((p) => p.storageKey),
    );
  }
  const result = await db.user.deleteMany({ where });
  console.log(
    `已清理本次 E2E 的 ${result.count} 個測試帳號與 ${photos.length} 張照片。`,
  );
} finally {
  await db.$disconnect();
}
