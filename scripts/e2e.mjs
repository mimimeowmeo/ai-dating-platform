import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
const env = { ...process.env, E2E_RUN_ID: randomUUID() };
function run(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { env, stdio: "inherit" });
    child.once("error", (error) => {
      console.error(error.message);
      resolve(1);
    });
    child.once("exit", (code) => resolve(code ?? 1));
  });
}
const code = await run("pnpm", [
  "--filter",
  "@dating/web",
  "test:e2e",
  ...process.argv.slice(2),
]);
const cleanup = await run("node", ["apps/api/test/cleanup-e2e.mjs"]);
process.exitCode = code || cleanup;
