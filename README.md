# 🎨 個人色彩分析器 (Personal Color Analyzer)

一個基於 AI 臉部偵測技術的個人色彩分析系統。本專案使用 **dlib 68 點臉部特徵點** 進行高精度的臉部偵測和色彩分析，提供專業級的個人色彩分析服務。

## ✨ 主要功能

- **高精度臉部偵測**: 使用 dlib 68 點臉部特徵點，精準勾勒五官與臉部輪廓
- **多區域顏色分析**: 自動從臉部區域中提取皮膚、眼睛、頭髮的代表性顏色
- **季節性色彩理論**: 基於 HSV 色彩理論，科學地分析您的個人色彩季節（春/夏/秋/冬）
- **AI 智能建議**: 整合 Google Gemini AI，根據您的分析結果，提供客製化的服裝、彩妝與飾品建議
- **視覺化偵錯**: 產生標示出 AI 辨識區域的偵錯圖，讓分析過程一目了然

## 🚀 快速開始

### 系統需求
- Python 3.8+
- Node.js 16+
- Windows 10/11

### 一鍵啟動
```bash
# 在專案根目錄，直接執行此批次檔即可啟動所有服務
start-all.bat
```
腳本會自動安裝所有必要的依賴（包括 dlib），並啟動前端、後端與 Python 分析器服務。

---

## 🛠️ 技術特色

本專案使用 **dlib** 進行高精度的臉部特徵點偵測，相比傳統的 OpenCV Haar 分類器，dlib 能夠：

- **精確定位 68 個臉部特徵點**
- **更準確的臉部輪廓識別**
- **更精細的區域分割**（膚色、眼睛、頭髮）
- **更穩定的分析結果**

系統會在首次啟動時自動下載 dlib 所需的模型檔案，無需手動配置。

---

## 📁 專案結構

```
personal-color-analyzer/
├── analyzer/
│   ├── app.py               # dlib 分析器（主要）
│   ├── app_opencv_only.py   # OpenCV 分析器（備用）
│   ├── download_models.py   # dlib 模型下載腳本
│   └── requirements.txt     # Python 依賴
├── backend/
├── frontend/
└── start-all.bat
```

## 🔧 技術架構

- **分析器 (Python + Flask)**:
  - **dlib**: 專業的臉部特徵點識別
  - **OpenCV**: 圖像處理和色彩分析
  - **Scikit-learn**: 用於 K-Means 顏色聚類分析
- **後端 (Node.js + Express)**: RESTful API，整合 Google Gemini AI
- **前端 (React + Vite)**: 現代化的使用者介面

## 🎯 分析流程

1. **圖片上傳** → 前端接收用戶圖片
2. **臉部偵測** → dlib 識別臉部特徵點
3. **區域提取** → 分別提取膚色、眼睛、頭髮區域
4. **色彩分析** → 使用 K-means 聚類分析主要顏色
5. **季節判斷** → 基於 HSV 色彩理論判斷季節型
6. **AI 建議** → Gemini AI 生成專業建議

## 🌟 特色功能

### 精確的臉部區域識別
- **膚色區域**：臉頰 + 額頭區域
- **眼睛區域**：左右眼虹膜區域
- **頭髮區域**：臉部上方和側面頭髮

### 智能備用機制
- 當臉部偵測失敗時，自動使用傳統方法
- 確保系統的穩定性和可靠性

### 專業的色彩建議
- 服裝顏色建議
- 彩妝顏色建議  
- 珠寶配飾建議
- 避免顏色提醒

## 📊 API 端點

### 色彩分析
```
POST /analyze
Content-Type: multipart/form-data
Body: { file: image_file }
```

回應格式：
```json
{
  "skin_tone": "#FFDAB9",
  "eye_color": "#8B4513",
  "hair_color": "#A0522D", 
  "season": "autumn",
  "season_name": "秋季型",
  "color_suggestions": {
    "clothes": ["#8B4513", "#A0522D", "#CD853F"],
    "makeup": ["#DEB887", "#F4A460", "#DAA520"],
    "jewelry": ["#B8860B", "#DAA520", "#CD853F"],
    "avoid": ["#00CED1", "#40E0D0", "#48D1CC"]
  }
}
```

### AI 建議
```
POST /api/gemini-suggestion
Content-Type: application/json
Body: { season, season_name, skin_tone, eye_color, hair_color, color_suggestions }
```

## 🔑 環境變數

在 `backend` 目錄創建 `.env` 文件：
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## 📝 使用說明

1. **上傳圖片**：拖拽或點擊上傳清晰的正面照片
2. **等待分析**：系統自動進行臉部偵測和色彩分析
3. **查看結果**：獲得膚色、眼睛、頭髮顏色分析
4. **季節判斷**：系統自動判斷您的季節型
5. **AI 建議**：獲得專業的個人化色彩建議

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

## 🙏 致謝

- [dlib](http://dlib.net/) - 臉部偵測和特徵點識別
- [OpenCV](https://opencv.org/) - 圖像處理
- [Google Gemini AI](https://ai.google.dev/) - AI 建議生成
- [React](https://reactjs.org/) - 前端框架
- [Flask](https://flask.palletsprojects.com/) - Python Web 框架 