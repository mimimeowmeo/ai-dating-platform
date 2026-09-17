import { existsSync, readFileSync } from "node:fs";
import { parseEnv } from "node:util";
import { spawn } from "node:child_process";
const path = new URL("../.env", import.meta.url);
const env = {
  ...(existsSync(path) ? parseEnv(readFileSync(path, "utf8")) : {}),
  ...process.env,
};
const [command, ...args] = process.argv.slice(2);
if (!command) throw new Error("請提供執行指令。");
const child = spawn(command, args, { env, stdio: "inherit" });
child.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
child.on("exit", (code) => {
  process.exitCode = code ?? 1;
});
for (const signal of ["SIGINT", "SIGTERM"])
  process.on(signal, () => child.kill(signal));
