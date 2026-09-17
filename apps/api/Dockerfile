FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN npm install -g pnpm@11.19.0
WORKDIR /app
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY apps/api/package.json ./apps/api/package.json
COPY apps/api/prisma ./apps/api/prisma
RUN pnpm install --frozen-lockfile --filter @dating/api...
COPY apps/api ./apps/api
RUN pnpm --filter @dating/api build
WORKDIR /app/apps/api
USER node
EXPOSE 3001
CMD ["sh", "-c", "pnpm db:migrate && node dist/main.js"]
