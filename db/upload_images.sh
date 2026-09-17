#!/usr/bin/env bash
set -euo pipefail

# 圖片來源目錄；預設在本目錄的 seed/images，可用 IMAGES_DIR 覆寫成資料集所在位置。
IMAGES_DIR="${IMAGES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/seed/images}"
S3_ENDPOINT="${S3_ENDPOINT:-http://host.docker.internal:9000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-dating-local}"
S3_SECRET_KEY="${S3_SECRET_KEY:?請用環境變數提供 S3_SECRET_KEY}"
BUCKET="${BUCKET:-heartlink-media}"
PREFIX="${PREFIX:-profiles}"

docker run --rm -v "${IMAGES_DIR}:/imgs:ro" minio/mc sh -c "
  mc alias set hl '${S3_ENDPOINT}' '${S3_ACCESS_KEY}' '${S3_SECRET_KEY}'
  mc mb -p hl/${BUCKET}
  mc anonymous set download hl/${BUCKET}
  mc mirror --overwrite /imgs hl/${BUCKET}/${PREFIX}
"
echo "完成：已上傳到 ${BUCKET}/${PREFIX}/  （例如 ${S3_ENDPOINT}/${BUCKET}/${PREFIX}/163710.jpg）"
