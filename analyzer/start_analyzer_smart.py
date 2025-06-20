#!/usr/bin/env python3
"""
智能個人色彩分析器啟動腳本
自動選擇最佳的分析模式
"""

import os
import sys
import subprocess

def main():
    """
    智能啟動腳本
    檢查 dlib 是否可用，然後啟動對應的分析器應用。
    """
    print("="*50)
    print("🚀 正在啟動個人色彩分析器 (智慧型選擇模式)...")
    
    app_to_run = "app_opencv_only.py" # 預設使用穩定版
    
    try:
        # 嘗試安靜地導入 dlib，不顯示任何輸出
        subprocess.check_output(
            [sys.executable, '-c', 'import dlib'],
            stderr=subprocess.STDOUT
        )
        # 如果上面的命令成功執行，表示 dlib 可用
        print("✅ dlib 模組可用，將啟動進階模式 (app.py)。")
        if os.path.exists("app.py"):
            app_to_run = "app.py"
        else:
            print("⚠️  進階模式檔案 (app.py) 不存在，退回標準模式。")

    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果導入失敗，則退回標準模式
        print("ℹ️  dlib 模組不可用或安裝不完整，將啟動標準模式 (app_opencv_only.py)。")
        app_to_run = "app_opencv_only.py"

    print(f"🎯 最終執行分析腳本: {app_to_run}")
    print("="*50)

    try:
        # 使用 execvp 來讓子進程完全取代當前進程
        # 這使得 Flask 的信號處理 (如 Ctrl+C) 更可靠
        os.execvp(sys.executable, [sys.executable, app_to_run])
    except Exception as e:
        print(f"❌ 啟動分析器 '{app_to_run}' 失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 