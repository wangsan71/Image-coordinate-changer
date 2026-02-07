# Photo Labeller Pro v3.0

這是一個基於 Python, OpenCV, 和 cvui 開發的高效率影像標註工具。它專為快速標定影像座標點而設計，並內建了 PyTorch AI 模型預測推論功能，能輔助使用者自動預測標註位置。

## 🌟 主要功能

- **雙模式介面**：
    - **設置模式**：直觀選擇來源資料夾與存放資料夾，並可調整 UI 顯示比例以適配不同解析度的螢幕。
    - **標註模式**：側邊欄控制面板搭配右側大圖顯示，支援滑鼠直覺點擊標註。
- **AI 輔助標註**：支援載入 `.pth` 模型（ResNet18 架構），開啟 **Auto Predict** 後，切換圖片時 AI 會自動給出建議標註點。
- **智慧導覽限制**：
    - 當處於第一張或最後一張圖片時，對應的導覽按鈕（Prev/Next）會自動**停用（Disable）並變灰**，防止索引溢位。
    - 點擊停用按鍵時，系統狀態列會顯示提示語。
- **快速鍵支援**：
    - `W` 或 `空白鍵`：儲存並下一張。
    - `A`：回上一張。
    - `D`：跳下一張。
    - `ESC`：退出程式。
- **自動命名存檔**：存檔時自動將標註座標 (x, y) 與隨機 UID 寫入檔名，方便後續訓練使用。

## 🛠️ 環境需求

- **Python 3.10+**
- **CUDA Toolkit 12.x** (建議針對 RTX 40 系列顯卡進行優化)
- **必要套件**：
  ```bash
  pip install opencv-python numpy cvui PyQt5 torch torchvision
  ```
🚀 快速啟動
執行主程式：

```Bash
python Photo_Labeller_Pro.py
```
在初始畫面選擇 Source Folder（原始圖片）與 Target Folder（儲存位置）。

點擊 START LABELLING 進入標註介面。
---
