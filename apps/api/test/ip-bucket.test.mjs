import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
const { ipBucket } = require("../src/ip-bucket.ts");
test("IPv4 與缺值維持原樣", () => {
  assert.equal(ipBucket("203.0.113.7"), "203.0.113.7");
  assert.equal(ipBucket(undefined), "unknown");
  assert.equal(ipBucket(""), "unknown");
  assert.equal(ipBucket("not-an-ip"), "not-an-ip");
});
test("同一段 IPv6 /56 共用一桶，不同 /56 分開", () => {
  const bucket = "2001:db8:1:200::/56";
  for (const ip of [
    "2001:db8:1:200::1",
    "2001:db8:1:2ff:ffff:ffff:ffff:ffff",
    "2001:0DB8:0001:0234:0000:0000:0000:0001",
  ])
    assert.equal(ipBucket(ip), bucket, ip);
  assert.equal(ipBucket("2001:db8:1:300::1"), "2001:db8:1:300::/56");
  assert.equal(ipBucket("2001:db8:2:200::1"), "2001:db8:2:200::/56");
});
test("IPv6 的各種寫法都能展開", () => {
  assert.equal(ipBucket("::1"), "0:0:0:0::/56");
  assert.equal(ipBucket("::"), "0:0:0:0::/56");
  assert.equal(ipBucket("2001:db8::"), "2001:db8:0:0::/56");
  assert.equal(ipBucket("fe80::1%eth0"), "fe80:0:0:0::/56");
  assert.equal(ipBucket("64:ff9b::192.0.2.1"), "64:ff9b:0:0::/56");
});
test("IPv4-mapped 位址還原成 IPv4", () => {
  assert.equal(ipBucket("::ffff:203.0.113.7"), "203.0.113.7");
  assert.equal(ipBucket("::FFFF:cb00:7107"), "203.0.113.7");
});
