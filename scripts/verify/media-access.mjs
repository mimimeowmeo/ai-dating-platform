// 照片存取控制：簽章網址才讀得到，竄改／過期／被封鎖都要擋下來。
// 用法：node scripts/verify/media-access.mjs
import { call, checker, makeAccount } from "./lib.mjs";

const check = checker("照片存取控制");
const A = await makeAccount("media-a", {
  displayName: "媒體甲",
  gender: "woman",
});
const B = await makeAccount("media-b", {
  displayName: "媒體乙",
  gender: "man",
});
const url = A.photo.url;

check(
  "上傳後回傳的網址帶簽章",
  /\/api\/v1\/media\/[0-9a-f-]+\?u=[0-9a-f-]+&e=\d+&s=[0-9a-f]{32}$/.test(url),
  url.slice(0, 72) + "…",
);
check(
  "自己的照片讀得到",
  (await call(url.replace("/api/v1", ""), { raw: true })).status === 200,
);
check(
  "沒有簽章讀不到",
  (await call(`/media/${A.photo.id}`, { raw: true })).status === 403,
);
check(
  "簽章被竄改讀不到",
  (
    await call(url.replace("/api/v1", "").replace(/s=.{4}/, "s=dead"), {
      raw: true,
    })
  ).status === 403,
);
check(
  "換成別人的 id 讀不到",
  (
    await call(
      url.replace("/api/v1", "").replace(/u=[0-9a-f-]+/, `u=${B.id}`),
      { raw: true },
    )
  ).status === 403,
);
check(
  "過期的網址讀不到",
  (
    await call(url.replace("/api/v1", "").replace(/e=\d+/, "e=1000000000000"), {
      raw: true,
    })
  ).status === 403,
);

// 探索卡片上的照片：給 A 看的網址，A 讀得到
const discovery = await call("/discovery", { token: A.token });
const card = (discovery.data || []).find((c) => c.userId === B.id);
check("探索卡片帶出對方的照片網址", !!card?.photos?.[0]?.url);
const seen = card.photos[0].url.replace("/api/v1", "");
check(
  "可以讀取探索卡片上的照片",
  (await call(seen, { raw: true })).status === 200,
);

// 封鎖之後，原本那組還沒過期的網址要立刻失效
await call("/blocks", {
  method: "POST",
  token: B.token,
  body: { blockedUserId: A.id },
});
check(
  "被對方封鎖後同一組網址失效",
  (await call(seen, { raw: true })).status === 404,
);
await call(`/blocks/${A.id}`, { method: "DELETE", token: B.token });
check("解除封鎖後恢復", (await call(seen, { raw: true })).status === 200);

const result = check.summary();
process.exitCode = result.failed ? 1 : 0;
