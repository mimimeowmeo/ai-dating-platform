FROM adminer:4
# 把自動登入的 index.php 烤進映像，取代預設首頁。
# 用 COPY 而非 bind mount：單檔掛載在來源檔被整個取代時（格式化/移除註解/git 操作）會失效，烤進映像可根治。
COPY index.php /var/www/html/index.php
