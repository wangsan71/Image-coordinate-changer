---

# Photo Labeller Pro v3.0

這是一個基於 **Python**, **OpenCV**, 和 **cvui** 開發的高效率影像標註工具。它專為快速標定影像座標點而設計，並內建了 **PyTorch** AI 模型的預測推論功能，能輔助使用者自動預測標註位置。

## 🌟 主要功能

* **雙模式介面**：
* **設置模式**：直觀選擇來源資料夾與存放資料夾，並可調整 UI 顯示比例以適配不同解析度的螢幕。
* **標註模式**：側邊欄控制面板搭配右側大圖顯示，支援滑鼠直覺點擊標註。


* **AI 輔助標註**：支援載入 `.pth` 模型（ResNet18 架構），開啟 **Auto Predict** 後，切換圖片時 AI 會自動給出建議標註點。
* **智慧導覽限制**：
* 當處於第一張或最後一張圖片時，對應的導覽按鈕（Prev/Next）會自動**停用（Disable）並變灰**，防止索引溢位。
* 點擊停用按鍵時，系統狀態列會顯示提示語。


* **快速鍵支援**：
* `W` 或 `空白鍵`：儲存並下一張。
* `A`：回上一張。
* `D`：跳下一張。
* `ESC`：退出程式。


* **自動命名存檔**：存檔時自動將標註座標（x, y）與隨機 UID 寫入檔名，方便後續訓練使用。

## 🛠️ 環境需求

* **Python 3.10+**
* **CUDA Toolkit 12.x** (建議針對 RTX 40 系列顯卡進行優化)
* **必要套件**：
```bash
pip install opencv-python numpy cvui PyQt5 torch torchvision

```



## 🚀 快速啟動

1. 執行主程式：
```bash
python Photo_Labeller_Pro.py

```


2. 在初始畫面選擇 **Source Folder**（原始圖片）與 **Target Folder**（儲存位置）。
3. 點擊 **START LABELLING** 進入標註介面。
4. （選配）點擊 **Load .pth Model** 載入預訓練模型以啟動 AI 輔助。

## ⚠️ 常見問題：WinError 1114 (DLL 載入失敗)

如果你在使用高性能筆電（如 ASUS、MSI）或 CUDA 12.8 環境時遇到 `c10.dll` 載入失敗問題，請參考以下修正方案：

1. **環境變數設定**：程式碼頂端已加入 `os.add_dll_directory` 指向 CUDA bin 目錄。請確保路徑正確：
```python
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

```


2. **顯卡偏好設置**：請至 Windows「圖形設定」中，將 `python.exe` 設置為「高效能（NVIDIA GPU）」。
3. **運行庫更新**：確保已安裝最新版的 [Microsoft Visual C++ Redistributable](https://www.google.com/search?q=https://aka.ms/vs/17/release/vc_redist.x64.exe)。

## 📝 授權與貢獻

本工具供個人及研究使用。如有功能建議或 Bug回報，歡迎隨時提出。

---
