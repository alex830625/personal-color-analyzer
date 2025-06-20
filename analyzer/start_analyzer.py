#!/usr/bin/env python3
"""
個人色彩分析器啟動腳本
自動下載必要模型並啟動服務
"""

import os
import sys
import subprocess
import time

def check_dlib_model():
    """檢查 dlib 模型文件是否存在"""
    return os.path.exists("shape_predictor_68_face_landmarks.dat")

def download_dlib_model():
    """下載 dlib 模型文件"""
    print("🔍 檢查 dlib 臉部特徵點模型...")
    
    if check_dlib_model():
        print("✅ 模型文件已存在")
        return True
    
    print("📥 模型文件不存在，開始下載...")
    try:
        # 執行下載腳本
        result = subprocess.run([sys.executable, "download_models.py"], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ 模型下載成功")
            return True
        else:
            print(f"❌ 模型下載失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 下載過程出錯: {e}")
        return False

def install_requirements():
    """安裝依賴套件"""
    print("📦 檢查並安裝依賴套件...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依賴套件安裝完成")
            return True
        else:
            print(f"❌ 依賴套件安裝失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安裝過程出錯: {e}")
        return False

def start_analyzer():
    """啟動分析器服務"""
    print("🚀 啟動個人色彩分析器...")
    try:
        # 啟動 Flask 應用
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 服務已停止")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")

def main():
    print("🎨 個人色彩分析器啟動程序")
    print("=" * 50)
    
    # 檢查並安裝依賴
    if not install_requirements():
        print("❌ 無法安裝依賴套件，程序退出")
        sys.exit(1)
    
    # 檢查並下載模型
    if not download_dlib_model():
        print("❌ 無法下載模型文件，程序退出")
        sys.exit(1)
    
    print("✅ 所有準備工作完成！")
    print("=" * 50)
    
    # 啟動服務
    start_analyzer()

if __name__ == "__main__":
    main() 