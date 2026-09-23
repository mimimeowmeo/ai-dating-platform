# 取自 minivision-ai/Silent-Face-Anti-Spoofing 的 src/model_lib/MiniFASNet.py（commit b6d5f04）。
# Copyright 2020 Minivision，依 Apache License 2.0 授權，全文見 third_party/licenses/。
# 本專案的修改：加上中文註解說明；模型定義與程式邏輯原樣保留（權重的 key 與層結構必須完全對得上）。
# 只在 Docker build 的 models 階段用來把 .pth 轉成 ONNX，不會進執行期映像。
# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午4:45
# @Author : zhuying
# @Company : Minivision
# @File : MiniFASNet.py
# @Software : PyCharm
# （以上 @ 開頭的幾行是上游作者原本的檔頭，原樣保留。
#   上面那行 `-*- coding: utf-8 -*-` 是 Python 的原始碼編碼宣告，依 PEP 263 只有放在檔案第 1 或第 2 行才有作用；
#   這裡因為前面加了授權說明而不在前兩行，但 Python 3 預設就用 UTF-8 讀原始碼，所以沒有影響。）
"""MiniFASNet 模型定義（PyTorch），供 export_minifasnet.py 載入 .pth 權重後轉成 ONNX。

這個檔案做什麼：
    用 PyTorch 的 Module 一層一層描述 MiniFASNet 的網路結構。PyTorch 的權重檔（.pth）只存「每一層的數值」，
    並不包含結構，所以要先用這裡的類別把同樣的結構建出來，再把權重依名稱（key）填進去。
    因此這裡的每個屬性名稱（例如 self.conv、self.bn、self.conv_6_dw）與每層的形狀都必須和上游訓練時一模一樣，
    改任何名稱或通道數都會讓 load_state_dict(strict=True) 失敗。

在本專案中的用途：
    export_minifasnet.py 只用到 MiniFASNetV2 與 MiniFASNetV1SE 兩個建構函式，
    呼叫時傳入 conv6_kernel=(5, 5)（配合 80×80 的輸入），轉成 ONNX 後由執行期的 onnxruntime 推論。
    模型輸出 3 個類別的分數（logits），執行期（app/models.py）取 index 1 當作「真人」。

網路整體結構（輸入 80×80 時的特徵圖大小）：
    conv1（stride 2，80→40）→ conv2_dw（40）→ conv_23（stride 2，40→20）→ conv_3（20）
    → conv_34（stride 2，20→10）→ conv_4（10）→ conv_45（stride 2，10→5）→ conv_5（5）
    → conv_6_sep（1×1 卷積，5）→ conv_6_dw（5×5 depthwise 卷積，5→1）→ 攤平成 512 維
    → linear（512→128）→ bn → drop → prob（128→類別數）。

常見名詞：
    - 卷積（convolution, Conv2d）：用小視窗在影像上滑動、計算特徵的層。
    - depthwise 卷積：groups 等於通道數時，每個通道各自獨立做卷積，參數量與運算量都比一般卷積少很多。
    - BatchNorm（批次正規化）：把每個通道的數值拉回穩定的分布，推論時使用訓練時累積的平均與變異數。
    - PReLU：帶可學習斜率的 ReLU 激活函式，負數部分乘上每個通道各自學到的係數。
    - 殘差連接（residual / shortcut）：把區塊的輸入直接加到輸出上，讓深層網路比較好訓練。
    - SE（Squeeze-and-Excitation）模組：先把每個通道壓成一個數字，再算出每個通道的權重去放大或縮小該通道。
"""
# PyTorch 主套件；這裡用到 torch.nn.Dropout。
import torch
# torch.nn.functional 提供不帶參數的函式版運算；這裡只在 L2Norm 用到 F.normalize。
import torch.nn.functional as F
# 從 torch.nn 匯入會用到的層：全連接層、2D 卷積、1D／2D 批次正規化、三種激活函式、自適應平均池化、
# 依序串接用的 Sequential，以及所有自訂層都要繼承的 Module 基底類別。
# 行尾的反斜線 `\` 表示這個 import 敘述接續到下一行。
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, \
    AdaptiveAvgPool2d, Sequential, Module


# L2 正規化層（上游定義；本檔案的模型都沒有用到它，原樣保留）。
class L2Norm(Module):
    """把輸入的每個向量縮放成長度（L2 範數）為 1。

    上游原始碼有定義這個類別，但 MiniFASNet／MiniFASNetSE 都沒有使用；
    為了讓這個檔案和上游保持一致，保留不刪。它沒有可學習的參數，不影響權重載入。
    """

    # PyTorch 的 Module 在被呼叫（layer(x)）時會執行 forward。
    def forward(self, input):
        """回傳沿著第 1 維（預設 dim=1）做 L2 正規化後的張量，形狀與輸入相同。

        參數：
            input：任意形狀至少 2 維的張量（名稱沿用上游，會遮蔽 Python 內建的 input 函式，但只在這個函式內有效）。

        回傳：
            正規化後的張量。
        """
        # F.normalize 預設 p=2、dim=1：每個樣本的向量除以自己的長度。
        return F.normalize(input)


# 攤平層：把 (批次, 通道, 高, 寬) 的特徵圖拉成 (批次, 通道×高×寬) 的向量，才能接全連接層。
class Flatten(Module):
    """把每個樣本的所有維度攤平成一維向量；在 MiniFASNet 中用於 conv_6_dw 之後（輸出 512 維）。"""

    # 前向運算。
    def forward(self, input):
        """參數 input：形狀 (N, C, H, W) 的張量。回傳：形狀 (N, C×H×W) 的張量。沒有可學習參數。"""
        # input.size(0) 是批次大小 N；-1 讓 PyTorch 自動算出剩下的長度。view 只改變形狀、不複製資料。
        return input.view(input.size(0), -1)


# 基本卷積區塊：卷積 → 批次正規化 → PReLU 激活。
class Conv_block(Module):
    """Conv2d + BatchNorm2d + PReLU 的組合，是這個網路最常用的積木。

    權重 key 會是 `<前綴>.conv.weight`、`<前綴>.bn.*`、`<前綴>.prelu.weight`，名稱不能改。
    """

    # 建立三個子層。
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        """參數：
            in_c／out_c：輸入／輸出通道數。
            kernel：卷積核大小（高, 寬），預設 1×1。
            stride：步幅；(2, 2) 會讓特徵圖的高寬減半。
            padding：邊緣補零的寬度；3×3 卷積配 padding (1, 1) 可維持高寬不變（stride 為 1 時）。
            groups：分組數；等於輸入通道數時就是 depthwise 卷積。

        設計理由：卷積後面緊接 BatchNorm，BatchNorm 自帶平移量，所以卷積不需要 bias（bias=False）。
        """
        # 呼叫父類別 Module 的初始化（舊式寫法，等同 super().__init__()），PyTorch 才能追蹤底下的子層與參數。
        super(Conv_block, self).__init__()
        # 2D 卷積層；bias=False 理由見 docstring。
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        # 對 out_c 個通道做批次正規化。
        self.bn = BatchNorm2d(out_c)
        # PReLU 激活，每個通道各有一個可學習的負半軸斜率（共 out_c 個參數）。
        self.prelu = PReLU(out_c)

    # 前向運算：依序通過三個子層。
    def forward(self, x):
        """參數 x：形狀 (N, in_c, H, W) 的張量。回傳：形狀 (N, out_c, H', W') 的張量（H'、W' 由 stride／padding 決定）。"""
        # 卷積。
        x = self.conv(x)
        # 批次正規化。
        x = self.bn(x)
        # 非線性激活。
        x = self.prelu(x)
        # 回傳結果。
        return x


# 線性卷積區塊：卷積 → 批次正規化，沒有激活函式。
class Linear_block(Module):
    """Conv2d + BatchNorm2d，刻意不加激活函式。

    用在 Depth_Wise 最後的 1×1 投影（project）與 conv_6_dw。投影層不加非線性激活，
    是 MobileNetV2 這類「linear bottleneck」結構的做法，可避免在通道數很少的地方用 ReLU 類函式丟失資訊。
    """

    # 建立兩個子層；參數意義同 Conv_block。
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        """參數同 Conv_block：in_c／out_c 通道數、kernel 卷積核、stride 步幅、padding 補零、groups 分組數。"""
        # 初始化父類別 Module。
        super(Linear_block, self).__init__()
        # 2D 卷積層；後面接 BatchNorm，所以不需要 bias。
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        # 批次正規化。
        self.bn = BatchNorm2d(out_c)

    # 前向運算。
    def forward(self, x):
        """參數 x：輸入張量。回傳：卷積 + 批次正規化後的張量（沒有激活）。"""
        # 卷積。
        x = self.conv(x)
        # 批次正規化。
        x = self.bn(x)
        # 回傳結果。
        return x


# 深度可分離卷積區塊：1×1 卷積 → depthwise 卷積 → 1×1 線性投影，可選擇加上殘差連接。
# （注意：上游這個類別的 def 用 5 個空白縮排，和其他類別不同；Python 只要求同一區塊內一致，所以能正常執行，原樣保留。）
class Depth_Wise(Module):
     """三段式卷積區塊，是 MiniFASNet 的主要構成單元。

     結構：conv（1×1 Conv_block，調整通道數）→ conv_dw（depthwise Conv_block，負責空間方向的特徵與降採樣）
     → project（1×1 Linear_block，投影到輸出通道數，不加激活）。
     residual=True 時把區塊輸入直接加到輸出（此時輸入與輸出的形狀必須相同，所以只用在 stride 1 的區塊）。

     參數：
         c1、c2、c3：三個 (輸入通道, 輸出通道) 的 tuple，分別給 conv、conv_dw、project 使用。
         residual：是否加殘差連接。
         kernel／stride／padding：depthwise 卷積的設定；預設 stride (2, 2) 會讓高寬減半。
         groups：上游保留的參數，實際上沒有使用（conv_dw 的分組數固定是 c2_in，也就是 depthwise）。
     """
     # 建立三個子層並記錄是否使用殘差連接。
     def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        """參數見類別說明。不回傳值；通道數不合理時由 PyTorch 在建立層時丟出錯誤。"""
        # 初始化父類別 Module。
        super(Depth_Wise, self).__init__()
        # 把第一段的 (輸入, 輸出) 通道拆成兩個變數。
        c1_in, c1_out = c1
        # 把第二段（depthwise）的通道拆開。
        c2_in, c2_out = c2
        # 把第三段（投影）的通道拆開。
        c3_in, c3_out = c3
        # 第一段：1×1 卷積 + BN + PReLU，只改變通道數，不改變高寬。
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 第二段：groups=c2_in 的 depthwise 卷積，每個通道各自做 kernel 大小的卷積；stride 為 2 時高寬減半。
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        # 第三段：1×1 線性投影到輸出通道數（沒有激活函式）。
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 記錄是否要加殘差連接，forward 時使用。
        self.residual = residual

     # 前向運算。
     def forward(self, x):
        """參數 x：輸入張量。回傳：三段卷積的結果；residual=True 時再加上原始輸入。"""
        # 需要殘差連接時，先把原始輸入存起來。
        if self.residual:
            # short_cut 保存進入區塊前的張量。
            short_cut = x
        # 第一段：1×1 卷積。
        x = self.conv(x)
        # 第二段：depthwise 卷積。
        x = self.conv_dw(x)
        # 第三段：1×1 線性投影。
        x = self.project(x)
        # 需要殘差連接時，把原始輸入與結果逐元素相加。
        if self.residual:
            # 殘差相加（兩者形狀必須相同）。
            output = short_cut + x
        # 不需要殘差連接時。
        else:
            # 直接使用三段卷積的結果。
            output = x
        # 回傳區塊輸出。
        return output


# 殘差區塊堆疊：把 num_block 個帶殘差連接的 Depth_Wise 依序串起來。
class Residual(Module):
    """num_block 個 Depth_Wise（residual=True）依序串接。

    參數：
        c1、c2、c3：長度為 num_block 的 list，第 i 個元素是第 i 個區塊的 (輸入, 輸出) 通道 tuple。
        num_block：區塊數。
        groups：傳給 Depth_Wise，但 Depth_Wise 並不使用它（見 Depth_Wise 的說明）。
        kernel／stride／padding：每個區塊 depthwise 卷積的設定；預設 stride (1, 1) 讓高寬不變，殘差相加才合法。

    權重 key 的形式是 `<前綴>.model.<第幾個區塊>.<子層>...`，所以屬性名稱 self.model 不能改。
    """

    # 建立 num_block 個區塊。
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        """參數見類別說明。"""
        # 初始化父類別 Module。
        super(Residual, self).__init__()
        # 暫存要串接的區塊。
        modules = []
        # 依序建立第 0 到第 num_block-1 個區塊。
        for i in range(num_block):
            # 第 i 個區塊的第一段通道設定。
            c1_tuple = c1[i]
            # 第 i 個區塊的 depthwise 通道設定。
            c2_tuple = c2[i]
            # 第 i 個區塊的投影通道設定。
            c3_tuple = c3[i]
            # 建立帶殘差連接的 Depth_Wise 並加入清單。
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        # 用 Sequential 串起來；`*modules` 把清單展開成多個參數。
        self.model = Sequential(*modules)

    # 前向運算。
    def forward(self, x):
        """參數 x：輸入張量。回傳：依序通過所有區塊後的張量（形狀和輸入相同）。"""
        # Sequential 會依序呼叫每個區塊。
        return self.model(x)


# Squeeze-and-Excitation（SE）模組：替每個通道算一個 0–1 的權重，再乘回特徵圖。
class SEModule(Module):
    """通道注意力模組。

    流程：全域平均池化把每個通道壓成 1 個數（squeeze）→ 1×1 卷積把通道數縮小為 1/reduction → BN → ReLU
    → 1×1 卷積把通道數還原 → BN → Sigmoid 得到每個通道 0–1 的權重（excitation）→ 乘回原本的特徵圖。
    讓模型學會強調有用的通道、壓低不重要的通道。

    參數：
        channels：輸入（也是輸出）的通道數。
        reduction：中間層通道數的縮小倍數。
    """

    # 建立子層。
    def __init__(self, channels, reduction):
        """參數見類別說明。"""
        # 初始化父類別 Module。
        super(SEModule, self).__init__()
        # 自適應平均池化：不論輸入高寬多少，都輸出 1×1（每個通道取平均）。
        self.avg_pool = AdaptiveAvgPool2d(1)
        # 第一個 1×1 卷積（作用等同全連接層），通道數從 channels 縮小到 channels // reduction（// 是整數除法）。
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        # 對縮小後的通道做批次正規化。
        self.bn1 = BatchNorm2d(channels // reduction)
        # ReLU 激活；inplace=True 直接改寫輸入張量以節省記憶體。
        self.relu = ReLU(inplace=True)
        # 第二個 1×1 卷積，把通道數還原成 channels。
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        # 對還原後的通道做批次正規化。
        self.bn2 = BatchNorm2d(channels)
        # Sigmoid 把數值壓到 0–1，當作每個通道的權重。
        self.sigmoid = Sigmoid()

    # 前向運算。
    def forward(self, x):
        """參數 x：形狀 (N, C, H, W) 的張量。回傳：同形狀、每個通道乘上各自權重後的張量。"""
        # 保留原始輸入，最後要乘上權重。
        module_input = x
        # squeeze：每個通道平均成一個數，形狀變成 (N, C, 1, 1)。
        x = self.avg_pool(x)
        # 縮小通道數。
        x = self.fc1(x)
        # 批次正規化。
        x = self.bn1(x)
        # 非線性激活。
        x = self.relu(x)
        # 還原通道數。
        x = self.fc2(x)
        # 批次正規化。
        x = self.bn2(x)
        # 轉成 0–1 的通道權重。
        x = self.sigmoid(x)
        # excitation：(N, C, 1, 1) 的權重透過廣播乘到 (N, C, H, W) 的原始輸入上。
        return module_input * x


# 帶 SE 的殘差區塊堆疊：前面幾個是一般 Depth_Wise，最後一個換成 Depth_Wise_SE。
class ResidualSE(Module):
    """和 Residual 相同，但最後一個區塊使用 Depth_Wise_SE（多一個 SE 模組）。

    參數：
        c1、c2、c3、num_block、groups、kernel、stride、padding：同 Residual。
        se_reduct：SE 模組的縮小倍數，預設 4（MiniFASNetSE 呼叫時沒有另外指定，所以用 4）。
    """

    # 建立 num_block 個區塊。
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        """參數見類別說明。"""
        # 初始化父類別 Module。
        super(ResidualSE, self).__init__()
        # 暫存要串接的區塊。
        modules = []
        # 依序建立每個區塊。
        for i in range(num_block):
            # 第 i 個區塊的第一段通道設定。
            c1_tuple = c1[i]
            # 第 i 個區塊的 depthwise 通道設定。
            c2_tuple = c2[i]
            # 第 i 個區塊的投影通道設定。
            c3_tuple = c3[i]
            # 最後一個區塊（index 為 num_block-1）使用帶 SE 的版本。
            if i == num_block-1:
                # 加入帶 SE 模組、帶殘差連接的區塊。
                modules.append(
                    Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride,
                               groups=groups, se_reduct=se_reduct))
            # 其他區塊。
            else:
                # 加入一般的帶殘差連接區塊。
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding,
                                          stride=stride, groups=groups))
        # 用 Sequential 串起來（屬性名稱 model 與權重 key 對應，不能改）。
        self.model = Sequential(*modules)

    # 前向運算。
    def forward(self, x):
        """參數 x：輸入張量。回傳：依序通過所有區塊後的張量。"""
        # 依序通過每個區塊。
        return self.model(x)


# 帶 SE 模組的 Depth_Wise：三段卷積之後、殘差相加之前先經過 SE。
# （這個類別在 ResidualSE 中被引用，但定義在 ResidualSE 之後；Python 在「呼叫」時才查名稱，所以沒有問題。）
class Depth_Wise_SE(Module):
    """結構與 Depth_Wise 相同，另外在 project 之後加上 SEModule。

    參數：
        c1、c2、c3、residual、kernel、stride、padding、groups：同 Depth_Wise（groups 同樣沒有使用）。
        se_reduct：SE 模組的縮小倍數；這裡預設 8，但 ResidualSE 一律會傳入自己的 se_reduct（預設 4）。
    """

    # 建立子層。
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        """參數見類別說明。"""
        # 初始化父類別 Module。
        super(Depth_Wise_SE, self).__init__()
        # 拆開第一段的 (輸入, 輸出) 通道。
        c1_in, c1_out = c1
        # 拆開第二段（depthwise）的通道。
        c2_in, c2_out = c2
        # 拆開第三段（投影）的通道。
        c3_in, c3_out = c3
        # 第一段：1×1 卷積 + BN + PReLU。
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 第二段：depthwise 卷積。
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        # 第三段：1×1 線性投影。
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 記錄是否使用殘差連接。
        self.residual = residual
        # SE 模組，作用在投影後的 c3_out 個通道上。
        self.se_module = SEModule(c3_out, se_reduct)

    # 前向運算。
    def forward(self, x):
        """參數 x：輸入張量。回傳：三段卷積結果；residual=True 時先經過 SE 再加上原始輸入。

        注意：依上游寫法，residual=False 時 SE 模組不會被使用（只有殘差分支會經過 SE）。
        """
        # 需要殘差連接時先保存原始輸入。
        if self.residual:
            # 保存進入區塊前的張量。
            short_cut = x
        # 第一段：1×1 卷積。
        x = self.conv(x)
        # 第二段：depthwise 卷積。
        x = self.conv_dw(x)
        # 第三段：1×1 線性投影。
        x = self.project(x)
        # 有殘差連接時。
        if self.residual:
            # 先經過 SE 模組替每個通道加權。
            x = self.se_module(x)
            # 再與原始輸入相加。
            output = short_cut + x
        # 沒有殘差連接時。
        else:
            # 直接使用三段卷積的結果（不經過 SE）。
            output = x
        # 回傳區塊輸出。
        return output


# MiniFASNet 主模型（不帶 SE）。
class MiniFASNet(Module):
    """MiniFASNet 完整網路：一連串卷積區塊把 80×80 影像壓成 512 維向量，再用全連接層輸出各類別分數。

    參數：
        keep：長度 49 的通道數清單（來自 keep_dict），依序決定每一層的輸入／輸出通道數。
              權重的形狀就是依這些數字產生的，所以必須用和訓練時相同的清單。
        embedding_size：中間特徵向量的維度（本專案用預設 128）。
        conv6_kernel：conv_6_dw 的卷積核大小；必須等於最後特徵圖的高寬，80×80 輸入時是 (5, 5)
                      （預設 (7, 7) 對應 112×112 的輸入，export_minifasnet.py 會改傳 (5, 5)）。
        drop_p：Dropout 的機率；只在訓練模式有作用，eval 模式（推論與匯出時）不會丟棄任何值。
        num_classes：輸出類別數（本專案用 3）。
        img_channel：輸入影像的通道數（3，BGR）。
    """

    # 建立所有層。
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),
                 drop_p=0.0, num_classes=3, img_channel=3):
        """參數見類別說明。不回傳值。"""
        # 初始化父類別 Module。
        super(MiniFASNet, self).__init__()
        # 記下 embedding_size，forward 時用來決定要不要經過 self.linear。
        self.embedding_size = embedding_size

        # 第一層：3×3、stride 2 的卷積，從 img_channel 個通道變成 keep[0] 個，高寬減半（80→40）。
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        # 第二層：3×3 depthwise 卷積（groups=keep[1]），高寬不變。
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        # conv_23 的三段通道設定：(keep[1]→keep[2])、(keep[2]→keep[3])、(keep[3]→keep[4])。
        c1 = [(keep[1], keep[2])]
        # depthwise 段的通道設定。
        c2 = [(keep[2], keep[3])]
        # 投影段的通道設定。
        c3 = [(keep[3], keep[4])]

        # conv_23：stride 2 的 Depth_Wise（沒有殘差），高寬減半（40→20）。groups 參數在 Depth_Wise 中未使用。
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        # conv_3 的 4 個區塊的第一段通道設定（每個區塊用 keep 裡連續的 3 個通道數，前一個區塊的輸出就是下一個的輸入）。
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        # 4 個區塊的 depthwise 段通道設定。
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        # 4 個區塊的投影段通道設定。
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        # conv_3：4 個帶殘差的區塊，高寬不變（20）。
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv_34 的第一段通道設定。
        c1 = [(keep[16], keep[17])]
        # depthwise 段的通道設定。
        c2 = [(keep[17], keep[18])]
        # 投影段的通道設定。
        c3 = [(keep[18], keep[19])]

        # conv_34：stride 2 的 Depth_Wise，高寬減半（20→10）。
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        # conv_4 的 6 個區塊的第一段通道設定。
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        # 6 個區塊的 depthwise 段通道設定。
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        # 6 個區塊的投影段通道設定。
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]

        # conv_4：6 個帶殘差的區塊，高寬不變（10）。
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv_45 的第一段通道設定。
        c1 = [(keep[37], keep[38])]
        # depthwise 段的通道設定。
        c2 = [(keep[38], keep[39])]
        # 投影段的通道設定。
        c3 = [(keep[39], keep[40])]

        # conv_45：stride 2 的 Depth_Wise，高寬減半（10→5）。
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        # conv_5 的 2 個區塊的第一段通道設定。
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        # 2 個區塊的 depthwise 段通道設定。
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        # 2 個區塊的投影段通道設定。
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]

        # conv_5：2 個帶殘差的區塊，高寬不變（5）。
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        # conv_6_sep：1×1 卷積，把通道數從 keep[46] 擴大到 keep[47]（512）。
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # conv_6_dw：depthwise 線性卷積，卷積核等於特徵圖大小且不補零，把 5×5 壓成 1×1（每個通道剩一個數）。
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        # 攤平成 (N, 512) 的向量。
        self.conv_6_flatten = Flatten()
        # 全連接層：512 維 → embedding_size 維，不帶 bias（後面接 BatchNorm1d）。
        self.linear = Linear(512, embedding_size, bias=False)
        # 對 embedding 向量做 1D 批次正規化。
        self.bn = BatchNorm1d(embedding_size)
        # Dropout：訓練時隨機把部分數值設為 0 以防過擬合；eval 模式下不作用。
        self.drop = torch.nn.Dropout(p=drop_p)
        # 輸出層：embedding_size 維 → num_classes 個類別分數（logits），不帶 bias。
        self.prob = Linear(embedding_size, num_classes, bias=False)

    # 前向運算：依序通過所有層。
    def forward(self, x):
        """參數 x：形狀 (N, 3, 80, 80) 的 float 張量（本專案送入 0–255 的 BGR 像素，不另外正規化）。

        回傳：形狀 (N, num_classes) 的 logits（尚未經過 softmax）。
        """
        # 第一層卷積（80→40）。
        out = self.conv1(x)
        # depthwise 卷積（40）。
        out = self.conv2_dw(out)
        # 降採樣區塊（40→20）。
        out = self.conv_23(out)
        # 4 個殘差區塊（20）。
        out = self.conv_3(out)
        # 降採樣區塊（20→10）。
        out = self.conv_34(out)
        # 6 個殘差區塊（10）。
        out = self.conv_4(out)
        # 降採樣區塊（10→5）。
        out = self.conv_45(out)
        # 2 個殘差區塊（5）。
        out = self.conv_5(out)
        # 1×1 卷積擴大通道數到 512。
        out = self.conv_6_sep(out)
        # 5×5 depthwise 卷積把空間壓成 1×1。
        out = self.conv_6_dw(out)
        # 攤平成 (N, 512)。
        out = self.conv_6_flatten(out)
        # embedding_size 不是 512 時才需要全連接層降維（本專案是 128，所以會執行）。
        if self.embedding_size != 512:
            # 512 → embedding_size。
            out = self.linear(out)
        # 批次正規化。
        out = self.bn(out)
        # Dropout（eval 模式下原樣通過）。
        out = self.drop(out)
        # 輸出各類別的分數。
        out = self.prob(out)
        # 回傳 logits。
        return out


# 帶 SE 模組的 MiniFASNet：沿用 MiniFASNet 的結構，把三段殘差堆疊換成 ResidualSE。
class MiniFASNetSE(MiniFASNet):
    """MiniFASNet 的 SE 版本：conv_3、conv_4、conv_5 改用 ResidualSE（每段最後一個區塊帶 SE 模組），其餘層相同。

    參數同 MiniFASNet；預設 drop_p=0.75、num_classes=4，但本專案用的 MiniFASNetV1SE 會傳入 num_classes=3。
    """

    # 建立模型。
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),drop_p=0.75, num_classes=4, img_channel=3):
        """先用父類別建出完整的 MiniFASNet，再把三段 Residual 替換成 ResidualSE。

        設計理由：重新指定 self.conv_3 等屬性時，PyTorch 會用新的子模組取代舊的，
        所以最終模型（與權重 key）只包含 ResidualSE 版本。
        """
        # 呼叫父類別 MiniFASNet 的建構函式，建立所有層（包括稍後會被取代的 Residual）。
        super(MiniFASNetSE, self).__init__(keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                               drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)

        # conv_3 的 4 個區塊的第一段通道設定（與父類別相同）。
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        # depthwise 段通道設定。
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        # 投影段通道設定。
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        # 用 ResidualSE 取代父類別建立的 conv_3。
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv_4 的 6 個區塊的第一段通道設定（與父類別相同）。
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        # depthwise 段通道設定。
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        # 投影段通道設定。
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]

        # 用 ResidualSE 取代父類別建立的 conv_4。
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv_5 的 2 個區塊的第一段通道設定（與父類別相同）。
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        # depthwise 段通道設定。
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        # 投影段通道設定。
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        # 用 ResidualSE 取代父類別建立的 conv_5。
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))



# 兩組通道數設定，key 是上游取的名稱；每組 49 個數字，對應 MiniFASNet.__init__ 中 keep[0]～keep[48]。
# '1.8M' 給 MiniFASNetV1／MiniFASNetV1SE 使用，'1.8M_' 給 MiniFASNetV2／MiniFASNetV2SE 使用。
# 這些數字決定每層權重的形狀，必須和訓練時完全相同，不能修改。
keep_dict = {'1.8M': [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
                      64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
                      154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
                      52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
                      128, 26, 26, 128, 26, 26, 128, 512, 512],

             '1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                       13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                       52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                       128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
             }


# 以下四個函式是建構函式（factory）：選好通道設定，建立對應版本的模型。
# 下面這類 flops／params 註解是上游原有的（輸入 80×80 時的運算量與參數量，單位上游沒有註明）。
# (80x80) flops: 0.044, params: 0.41
def MiniFASNetV1(embedding_size=128, conv6_kernel=(7, 7),
                     drop_p=0.2, num_classes=3, img_channel=3):
    """建立 MiniFASNetV1（通道設定 '1.8M'、不帶 SE）。本專案沒有使用。參數同 MiniFASNet，回傳 MiniFASNet 實例。"""
    # 依位置把參數傳給 MiniFASNet。
    return MiniFASNet(keep_dict['1.8M'], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


# (80x80) flops: 0.044, params: 0.43
def MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7),
                     drop_p=0.2, num_classes=3, img_channel=3):
    """建立 MiniFASNetV2（通道設定 '1.8M_'、不帶 SE）。

    本專案搭配權重 2.7_80x80_MiniFASNetV2.pth 使用，export_minifasnet.py 只傳入 conv6_kernel=(5, 5)，
    其餘用預設值（embedding 128 維、3 個類別）。回傳 MiniFASNet 實例。
    """
    # 依位置把參數傳給 MiniFASNet。
    return MiniFASNet(keep_dict['1.8M_'], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)

# 建立 MiniFASNetV1SE（通道設定 '1.8M'、帶 SE）。
def MiniFASNetV1SE(embedding_size=128, conv6_kernel=(7, 7),
                   drop_p=0.75, num_classes=3, img_channel=3):
    """建立 MiniFASNetV1SE。

    本專案搭配權重 4_0_0_80x80_MiniFASNetV1SE.pth 使用，export_minifasnet.py 只傳入 conv6_kernel=(5, 5)。
    這裡的 num_classes 預設 3，會覆蓋 MiniFASNetSE 自己的預設 4。回傳 MiniFASNetSE 實例。
    """
    # 依位置把參數傳給 MiniFASNetSE。
    return MiniFASNetSE(keep_dict['1.8M'], embedding_size, conv6_kernel,drop_p, num_classes, img_channel)

# (80x80) flops: 0.044, params: 0.43
def MiniFASNetV2SE(embedding_size=128, conv6_kernel=(7, 7),
                   drop_p=0.75, num_classes=4, img_channel=3):
    """建立 MiniFASNetV2SE（通道設定 '1.8M_'、帶 SE、預設 4 個類別）。本專案沒有使用。回傳 MiniFASNetSE 實例。"""
    # 依位置把參數傳給 MiniFASNetSE。
    return MiniFASNetSE(keep_dict['1.8M_'], embedding_size, conv6_kernel,drop_p, num_classes, img_channel)
