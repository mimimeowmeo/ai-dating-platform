#!/usr/bin/env bash
# 重開 Cloudflare 臨時通道，並自動把 api 的 WEB_ORIGIN 對齊到新網址。
# 用法：bash scripts/retunnel.sh
# 停止：在本腳本視窗按 Ctrl+C（會自動關閉 tunnel）
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到專案根目錄

LOG="$(mktemp -t cf.XXXXXX)"
# 離開時（含 Ctrl+C）自動關閉 tunnel
trap 'echo; echo "→ 關閉 tunnel"; pkill -f "cloudflared tunnel" 2>/dev/null || true' EXIT

echo "→ 1/4 停掉舊 tunnel（如果有）"
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 2

echo "→ 2/4 啟動新 tunnel"
cloudflared tunnel --url http://localhost:8080 >"$LOG" 2>&1 &

echo "→ 3/4 等新網址（最多 60 秒）"
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -z "$URL" ] && { echo "✗ 逾時，沒抓到網址。原始輸出：$LOG"; exit 1; }
echo "  ✓ 新網址：$URL"

echo "→ 4/4 對齊 WEB_ORIGIN 並重啟 api"
WEB_ORIGIN="$URL" docker compose -f docker-compose.test.yml up -d --wait api

echo
echo "======================================================"
echo "  內部測試站已就緒 → $URL"
echo "  發給信任的內部人測。按 Ctrl+C 結束並自動關閉 tunnel。"
echo "======================================================"
wait   # 保持前景直到你 Ctrl+C；trap 會收拾 tunnel
