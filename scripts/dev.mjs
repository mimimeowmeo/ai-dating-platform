import { readFileSync } from "node:fs";
import { parseEnv } from "node:util";
import { spawn } from "node:child_process";
const env = {
  ...parseEnv(readFileSync(new URL("../.env", import.meta.url), "utf8")),
  ...process.env,
  NODE_ENV: "development",
  WEB_ORIGIN: "http://localhost:3000",
  API_PROXY_URL: "http://127.0.0.1:3001",
};
const children = ["@dating/api", "@dating/web"].map((name) =>
  spawn("pnpm", ["--filter", name, "dev"], { env, stdio: "inherit" }),
);
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach((child) => child.kill("SIGTERM"));
  process.exitCode = code;
}
children.forEach((child) => {
  child.on("error", (error) => {
    console.error(error.message);
    stop(1);
  });
  child.on("exit", (code) => stop(code ?? 1));
});
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
