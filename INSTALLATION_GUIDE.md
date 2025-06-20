# 🔧 安裝指南 - 個人色彩分析器

## 🚨 問題說明

dlib 在 Windows 上需要 Visual Studio C++ 編譯器才能編譯。如果您遇到編譯錯誤，請選擇以下解決方案之一。

## 🔧 解決方案

### 方案 1：使用智能啟動腳本（推薦）

我們已經創建了智能啟動腳本，會自動處理 dlib 安裝問題：

```bash
# 直接執行一鍵啟動
start-all.bat
```

智能腳本會：
1. 嘗試安裝 dlib
2. 如果失敗，自動切換到 OpenCV 模式
3. 確保系統正常運行

### 方案 2：手動安裝 Visual Studio Build Tools

如果您想要完整的 dlib 功能：

1. **下載 Visual Studio Build Tools**：
   - 前往 [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - 下載 "Build Tools for Visual Studio 2022"
   - 安裝時選擇 "C++ build tools"

2. **重新安裝 dlib**：
   ```bash
   cd analyzer
   pip install dlib
   ```

### 方案 3：使用預編譯的 dlib wheel

```bash
cd analyzer
pip install -r requirements.txt
```

我們已經在 requirements.txt 中配置了預編譯的 wheel。

### 方案 4：僅使用 OpenCV 模式

如果您不想安裝 dlib：

```bash
cd analyzer
pip install -r requirements_simple.txt
python app_opencv_only.py
```

## 📊 功能對比

| 功能 | dlib 模式 | OpenCV 模式 |
|------|-----------|-------------|
| 臉部偵測精度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 特徵點識別 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 膚色分析 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 眼睛顏色分析 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 頭髮顏色分析 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 安裝難度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚀 快速開始

### 第一次使用（推薦）

1. **執行智能啟動**：
   ```bash
   start-all.bat
   ```

2. **等待自動配置**：
   - 系統會自動嘗試安裝 dlib
   - 如果失敗，會自動切換到 OpenCV 模式
   - 所有服務會自動啟動

3. **開始使用**：
   - 前端：http://localhost:5173
   - 後端：http://localhost:3000
   - 分析器：http://localhost:5001

### 手動啟動

```bash
# 1. 啟動後端
cd backend
npm install
npm start

# 2. 啟動前端
cd frontend
npm install
npm run dev

# 3. 啟動分析器（智能模式）
cd analyzer
python start_analyzer_smart.py
```

## 🔍 故障排除

### 問題 1：dlib 編譯失敗

**解決方案**：
- 使用智能啟動腳本（推薦）
- 或安裝 Visual Studio Build Tools
- 或使用 OpenCV 模式

### 問題 2：模型下載失敗

**解決方案**：
- 檢查網路連接
- 手動下載模型文件
- 使用 OpenCV 模式（不需要模型）

### 問題 3：端口被佔用

**解決方案**：
```bash
# 查看端口使用情況
netstat -ano | findstr :5001

# 終止佔用端口的進程
taskkill /PID <進程ID> /F
```

## 📝 注意事項

1. **OpenCV 模式**：雖然精度稍低，但功能完整，適合大多數使用場景
2. **dlib 模式**：提供最高精度，但需要額外的編譯環境
3. **智能模式**：自動選擇最佳方案，推薦使用

## 🆘 需要幫助？

如果遇到問題：

1. 查看控制台錯誤訊息
2. 嘗試使用 OpenCV 模式
3. 檢查網路連接
4. 重新啟動所有服務

## ✅ 驗證安裝

啟動後，您可以：

1. 上傳一張清晰的正面照片
2. 查看分析結果
3. 確認所有功能正常運作

如果一切正常，您會看到：
- 膚色、眼睛、頭髮顏色分析
- 季節型判斷
- AI 生成的專業建議 