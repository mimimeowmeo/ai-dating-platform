import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
require("ts-node/register/transpile-only");
const { webOrigins } = require("../src/web-origins.ts");
test("loopback aliases retain the configured protocol and port", () => {
  assert.deepEqual(webOrigins("http://localhost:8080"), [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
  ]);
  assert.deepEqual(webOrigins("http://127.0.0.1:3000"), [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
  ]);
  const allowed = webOrigins("http://localhost:8080");
  for (const origin of [
    "http://localhost:8081",
    "https://localhost:8080",
    "http://localhost.evil.test:8080",
    "http://127.0.0.2:8080",
    "null",
  ])
    assert.equal(allowed.includes(origin), false);
});
test("deployed domains do not gain local or subdomain access", () => {
  assert.deepEqual(webOrigins("https://meet.example.test"), [
    "https://meet.example.test",
  ]);
});
