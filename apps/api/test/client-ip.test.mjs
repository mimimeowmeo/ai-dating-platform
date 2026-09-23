import test, { after } from "node:test";
import assert from "node:assert/strict";
import { randomInt } from "node:crypto";
import Redis from "ioredis";
const base = process.env.TEST_API_URL || "http://127.0.0.1:3001/api/v1";
const tunnel = process.env.NGINX_CLIENT_IP === "cloudflare-tunnel";
// 直連 API 埠時本來就能偽造 X-Forwarded-For（見 apps/api/src/main.ts），這裡只驗經過 nginx 的情況。
const viaNginx =
  (await fetch(`${base}/health`)).headers.get("server") === "nginx";
const skip = viaNginx ? false : "TEST_API_URL 沒有經過 nginx";
const redis = new Redis(process.env.REDIS_URL, { lazyConnect: true });
const used = [];
after(async () => {
  const keys = used.flatMap((ip) => [`rate:auth:${ip}`, `rate:refresh:${ip}`]);
  if (keys.length) await redis.del(...keys);
  redis.disconnect();
});
function fakeIp() {
  const ip = `198.${18 + randomInt(2)}.${randomInt(256)}.${randomInt(1, 255)}`;
  used.push(ip);
  return ip;
}
const viaCloudflare = (ip, spoofed) => ({
  "CF-Connecting-IP": ip,
  "X-Forwarded-For": `${spoofed}, ${ip}`,
});
// 註冊額度（每個 IP 20 次／5 分鐘）已經快被 integration.test.mjs 用完，
// 會落在測試機自己 IP 的請求改打 refresh（同樣以 req.ip 限流，60 次／分）。
async function login(headers) {
  const res = await fetch(`${base}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: "{}",
  });
  return res.status;
}
async function refresh(headers) {
  const res = await fetch(`${base}/auth/refresh`, { method: "POST", headers });
  return res.status;
}
test(
  "direct：偽造 CF-Connecting-IP、X-Forwarded-For、X-Real-IP 都換不到新的限流額度",
  { skip: skip || (tunnel && "NGINX_CLIENT_IP=cloudflare-tunnel") },
  async () => {
    const [cf, forwarded, real] = [fakeIp(), fakeIp(), fakeIp()];
    const status = await refresh({
      "CF-Connecting-IP": cf,
      "X-Forwarded-For": forwarded,
      "X-Real-IP": real,
    });
    assert.ok([401, 429].includes(status), `status ${status}`);
    for (const ip of [cf, forwarded, real])
      assert.equal(await redis.exists(`rate:refresh:${ip}`), 0, ip);
  },
);
test(
  "cloudflare-tunnel：每個 CF-Connecting-IP 各有自己的額度，一人用完不會擋到別人",
  { skip: skip || (!tunnel && "NGINX_CLIENT_IP 不是 cloudflare-tunnel") },
  async () => {
    const [a, b, spoofed] = [fakeIp(), fakeIp(), fakeIp()];
    for (let i = 0; i < 20; i++)
      assert.equal(await login(viaCloudflare(a, spoofed)), 400);
    assert.equal(await login(viaCloudflare(a, spoofed)), 429);
    assert.equal(await login(viaCloudflare(b, spoofed)), 400);
    assert.equal(await redis.get(`rate:auth:${a}`), "21");
    assert.equal(await redis.get(`rate:auth:${b}`), "1");
    assert.equal(await redis.exists(`rate:auth:${spoofed}`), 0);
  },
);
test(
  "cloudflare-tunnel：CF-Connecting-IP 不是合法位址時沿用連線位址",
  { skip: skip || (!tunnel && "NGINX_CLIENT_IP 不是 cloudflare-tunnel") },
  async () => {
    const spoofed = fakeIp();
    const status = await refresh({
      "CF-Connecting-IP": "not-an-ip",
      "X-Forwarded-For": spoofed,
    });
    assert.ok([401, 429].includes(status), `status ${status}`);
    assert.equal(await redis.exists("rate:refresh:not-an-ip"), 0);
    assert.equal(await redis.exists(`rate:refresh:${spoofed}`), 0);
  },
);
