# 人臉驗證 provider（services/face）的映像檔定義，採用多階段建置（multi-stage build）。
# - 第一階段 models：安裝 torch，下載模型並轉檔；這個階段的東西（torch、.pth 權重、下載暫存）不會進最終映像。
# - 第二階段（執行期）：只放執行需要的套件、轉好的 ONNX 模型與程式碼，映像小、攻擊面也小。
# 建置的 context 是專案根目錄（docker-compose.yml 的 build.context 是 .），所以下面的 COPY 路徑都以 services/face/ 開頭；
# 同目錄的 Dockerfile.dockerignore 只放行這個服務需要的檔案。

# 第一階段：下載模型並驗 sha256，把 MiniFASNet 轉成 ONNX；torch 只存在這個階段。
# FROM：以官方 Python 3.12 精簡版（slim）映像為基底；AS models 替這個階段命名，第二階段用 --from=models 取用它的產物。
FROM python:3.12-slim AS models

# ENV：設定環境變數。
# - PYTHONDONTWRITEBYTECODE=1：Python 不寫出 .pyc 快取檔，映像裡不留多餘檔案。
# - PYTHONUNBUFFERED=1：stdout／stderr 不緩衝，輸出立刻出現在 build／容器 log。
# - PIP_NO_CACHE_DIR=1：pip 不保留下載快取，減少映像層的大小。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
# WORKDIR：設定之後指令的工作目錄（不存在會自動建立）；這個階段的所有檔案都放在 /build。
WORKDIR /build
# COPY：只複製轉檔工具需要的套件清單（numpy、onnx、onnxruntime），到 /build/requirements-export.txt。
COPY services/face/requirements-export.txt ./requirements-export.txt
# RUN：安裝轉檔需要的套件。
# - 先從 PyTorch 官方的 CPU 套件庫（download.pytorch.org/whl/cpu）安裝固定版本的 torch 2.5.1：只轉檔不需要 GPU，CPU 版不含 CUDA，小很多。
# - 再依 requirements-export.txt 安裝其餘套件（版本都鎖定，build 結果可重現）。
# - --no-cache-dir：不留 pip 快取。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -r requirements-export.txt
# COPY：複製工具程式（fetch_models.py 下載模型、export_minifasnet.py 轉檔、minifasnet.py 模型定義）到 /build/tools。
# 放在安裝套件之後：只改工具程式時，上面安裝 torch 的那層可以沿用 Docker 快取，不必重裝。
COPY services/face/tools ./tools
# RUN：取得並準備模型檔，任一步失敗（例如 sha256 不符、轉檔前後輸出不一致）整個 build 就失敗。
# 1. fetch_models.py 把四個模型檔下載到 downloads/，並驗證檔案大小與 sha256（網址鎖定在特定 commit）。
# 2. 建立 models/ 資料夾，存放最終要進執行期映像的模型。
# 3. YuNet（人臉偵測）與 SFace（人臉特徵）本來就是 ONNX，直接搬到 models/。
# 4. 進入 tools/ 執行 export_minifasnet.py：把 downloads/ 裡兩個 MiniFASNet 的 .pth 權重轉成 ONNX 寫到 models/，
#    並檢查 onnxruntime 與 PyTorch 的輸出一致。
RUN python tools/fetch_models.py downloads \
    && mkdir models \
    && mv downloads/face_detection_yunet_2023mar.onnx downloads/face_recognition_sface_2021dec.onnx models/ \
    && cd tools && python export_minifasnet.py ../downloads ../models

# 第二階段：執行期映像，只有 OpenCV、onnxruntime 與轉好的模型。
# FROM：重新從乾淨的 python:3.12-slim 開始；第一階段裝的 torch 與下載暫存都不會帶過來。
FROM python:3.12-slim

# ENV：同第一階段，不寫 .pyc、輸出不緩衝（print 的 log 立刻可見）、pip 不留快取。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
# WORKDIR：執行期的工作目錄是 /app；程式預設的模型資料夾 /app/models 就在這裡（見 app/config.py）。
WORKDIR /app
# COPY：複製執行期的套件清單（FastAPI、uvicorn、OpenCV headless、onnxruntime 等，版本全部鎖定）。
COPY services/face/requirements.txt ./requirements.txt
# RUN：
# - 安裝執行期套件（不留 pip 快取）。
# - useradd 建立系統帳號 face（固定 UID 10001，並建立家目錄），後面用這個非 root 帳號執行服務，
#   萬一服務被攻破，攻擊者也不會拿到容器內的 root 權限。
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 --create-home face
# COPY --from=models：從第一階段只複製轉好的模型資料夾（YuNet、SFace 與兩個 MiniFASNet 的 ONNX）到 /app/models。
COPY --from=models /build/models ./models
# COPY：複製第三方授權全文（YuNet、SFace、Silent-Face-Anti-Spoofing 的授權），讓授權聲明隨映像一起散布。
COPY services/face/third_party ./third_party
# COPY --chown=face:face：複製 provider 程式碼到 /app/app，並把檔案擁有者設成 face 帳號。
# 程式碼最常變動，放在最後複製，改程式時前面的層（安裝套件、模型）都能沿用快取。
COPY --chown=face:face services/face/app ./app
# USER：之後的指令與容器執行時都改用 face 帳號（非 root）。
USER face
# EXPOSE：宣告服務監聽 8100 埠（僅是文件用途，實際對外開放由 docker-compose 的 ports 決定）。
EXPOSE 8100
# HEALTHCHECK：Docker 每 20 秒檢查一次服務是否健康。
# - --interval=20s：檢查間隔；--timeout=5s：單次檢查超過 5 秒算失敗；
# - --start-period=30s：啟動後 30 秒內的失敗不計入（留時間載入模型）；--retries=3：連續失敗 3 次才標記為 unhealthy。
# - 檢查方式：用 Python 內建的 urllib 呼叫 GET /health。模型沒載入時 /health 回 503，urlopen 會丟出例外，
#   python 以非 0 結束碼離開，Docker 就判定這次檢查失敗。（映像沒有 curl，所以用 python。）
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3)"
# CMD：容器啟動時執行的指令。用 uvicorn（ASGI 伺服器）載入 app/main.py 裡的 app 物件：
# - --host 0.0.0.0：監聽所有網路介面，讓同一個 Docker 網路裡的 ai 容器能連進來（只聽 127.0.0.1 的話容器外連不到）。
# - --port 8100：監聽埠，和 EXPOSE、HEALTHCHECK 一致。
# - --no-access-log：關閉每個請求的存取紀錄，只留程式自己印的固定代碼。
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--no-access-log"]
