@echo off
echo 🚀 啟動個人色彩分析器系統...
echo.

echo 📊 系統資訊：
echo - 記憶體使用量檢查...
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:table

echo.
echo 🔧 啟動後端服務...
cd backend
start "Backend Server" cmd /k "npm install && npm start"

echo ⏳ 等待後端啟動...
timeout /t 5 /nobreak > nul

echo 🎨 啟動前端服務...
cd ../frontend
start "Frontend Server" cmd /k "npm install && npm run dev"

echo ⏳ 等待前端啟動...
timeout /t 5 /nobreak > nul

echo 🤖 啟動 Python 分析器服務 (優化模式)...
cd ../analyzer

REM 設定環境變數以優化性能
set PYTHONOPTIMIZE=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

REM 啟動分析器時先啟動 conda 環境，再執行 analyzer.py
start "Python Analyzer (Optimized)" cmd /k "C:\Users\User\anaconda3\Scripts\activate.bat dlib_env && python analyzer.py"

echo.
echo ✅ 所有服務已啟動！
echo.
echo 🌐 服務網址：
echo - 前端: http://localhost:5173
echo - 後端: http://localhost:3000
echo - 分析器: http://localhost:5001 (優化模式)
echo.
echo 📈 性能優化功能：
echo - ✅ 模型快取機制
echo - ✅ 圖片尺寸優化
echo - ✅ 並行處理
echo - ✅ API 快取
echo - ✅ 多執行緒支援
echo.
echo 💡 使用提示：
echo - 首次分析可能需要 3-5 秒載入模型
echo - 後續分析將大幅提升至 1-2 秒
echo - 相同圖片的分析結果會被快取
echo.
echo 按任意鍵退出...
pause > nul