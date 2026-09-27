# 推薦 worker：新的主照片 → 外貌向量（見 README.md）。
# models 階段下載並驗證模型檔；執行期不連網（HF_HUB_OFFLINE）。
FROM python:3.12-slim AS models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY services/recommendation/tools ./tools
RUN python tools/fetch_models.py models

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY services/recommendation/requirements.txt services/recommendation/requirements-nodeps.txt ./
# MediaPipe 的原生函式庫要 libEGL.so.1、libGLESv2.so.2（只用 CPU，不會真的呼叫）；
# libegl1 強制相依 Mesa，所以只解出它的檔案，做法同 services/face/Dockerfile。
# torch 從 PyTorch 的 CPU 套件庫安裝，避免 x86_64 拉到含 CUDA 的版本。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgles2 \
    && cd /tmp && apt-get download libegl1 && dpkg-deb -x libegl1_*.deb / && rm libegl1_*.deb && cd /app \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps -r requirements-nodeps.txt \
    && useradd --system --uid 10001 --create-home rec
COPY --from=models /build/models ./models
COPY services/recommendation/third_party ./third_party
COPY services/recommendation/appearance ./appearance
COPY services/recommendation/app ./app
COPY services/recommendation/tests ./tests
USER rec
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "app.worker_health"]
CMD ["python", "-m", "app.worker"]
