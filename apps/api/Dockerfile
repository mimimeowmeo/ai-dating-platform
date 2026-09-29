FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN npm install -g pnpm@11.19.0
WORKDIR /app
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY apps/api/package.json ./apps/api/package.json
COPY apps/api/prisma ./apps/api/prisma
RUN pnpm install --frozen-lockfile --filter @dating/api...
# 依賴只在上一步裝一次。之後的 pnpm run（下一步 build、容器啟動時的 db:migrate）都關掉 pnpm 11 的
# 執行前依賴檢查（verifyDepsBeforeRun，預設判定過期就自動 pnpm install）。這個檢查沒套用
# pnpm-workspace.yaml 的 overrides，會把 multer（package.json ^2.0.2、lockfile 2.4.0）誤判成過期：
# COPY 進來的 package.json 比快取的安裝層新時（本機切過分支就會），build 會自動重裝並改寫 lockfile；
# 容器啟動時又判定一次、以 node 身分重裝，寫不進 root 的 .pnpm-store，api 就一直重啟。
ENV pnpm_config_verify_deps_before_run=false
COPY apps/api ./apps/api
RUN pnpm --filter @dating/api build
WORKDIR /app/apps/api
USER node
EXPOSE 3001
CMD ["sh", "-c", "pnpm db:migrate && node dist/main.js"]
