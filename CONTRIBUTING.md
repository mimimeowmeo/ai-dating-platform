# 開發方式

1. 先閱讀 README、docs/implementation/PHASE-1-7-CONTRACT.md 與架構 ADR。
2. 執行 pnpm setup、pnpm install、pnpm infra:up。
3. 執行 pnpm db:migrate，再使用 pnpm dev。（AI 服務已移除，真人驗證會回 status=unavailable，屬預期行為。）
4. 變更請附上 typecheck、build 與對應 API／AI／Playwright 測試結果。
5. 不提交 .env，不使用 docker compose down -v 清除已有資料；migration 不得任意 reset。
6. API controller 只負責 HTTP 轉接，資料存取與業務規則集中在 service。
7. 模型、門檻與資料政策需明確決策，不得用假的成功回應通過驗證。
