@echo off
echo 啟動個人色彩分析器系統...
echo.

echo 啟動後端服務...
cd backend
start "Backend Server" cmd /k "npm install && npm start"

echo 等待後端啟動...
timeout /t 3 /nobreak > nul

echo 啟動前端服務...
cd ../frontend
start "Frontend Server" cmd /k "npm install && npm run dev"

echo 等待前端啟動...
timeout /t 3 /nobreak > nul

echo 啟動 Python 分析器服務 (dlib 模式)...
cd ../analyzer

REM 啟動分析器時先啟動 conda 環境，再執行 app.py
start "Python Analyzer (dlib)" cmd /k "C:\Users\User\anaconda3\Scripts\activate.bat dlib_env && python analyzer.py"

echo.
echo 所有服務已啟動！
echo 前端: http://localhost:5173
echo 後端: http://localhost:3000
echo 分析器: http://localhost:5001 (dlib 模式)
echo.
echo 按任意鍵退出...
pause > nul