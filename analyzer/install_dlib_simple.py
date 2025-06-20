#!/usr/bin/env python3
"""
簡化的 dlib 安裝腳本
嘗試多種方法安裝 dlib，避免編譯問題
"""

import subprocess
import sys
import os

def try_install_dlib():
    """嘗試安裝 dlib"""
    print("🔧 嘗試安裝 dlib...")
    
    # 方法 1：嘗試使用 conda（如果可用）
    try:
        result = subprocess.run([sys.executable, "-m", "conda", "install", "dlib", "-y"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ 使用 conda 安裝 dlib 成功")
            return True
    except:
        pass
    
    # 方法 2：嘗試使用預編譯的 wheel
    wheel_urls = [
        "https://github.com/sachadee/Dlib/releases/download/v19.24/dlib-19.24.2-cp311-cp311-win_amd64.whl",
        "https://github.com/sachadee/Dlib/releases/download/v19.24/dlib-19.24.2-cp310-cp310-win_amd64.whl",
        "https://github.com/sachadee/Dlib/releases/download/v19.24/dlib-19.24.2-cp39-cp39-win_amd64.whl"
    ]
    
    for url in wheel_urls:
        try:
            print(f"📥 嘗試下載: {url}")
            result = subprocess.run([sys.executable, "-m", "pip", "install", url], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✅ 使用預編譯 wheel 安裝 dlib 成功")
                return True
            else:
                print(f"⚠️  下載失敗: {result.stderr}")
        except Exception as e:
            print(f"⚠️  嘗試失敗: {e}")
    
    # 方法 3：嘗試使用 pip 安裝（可能會失敗）
    try:
        print("🔧 嘗試使用 pip 安裝 dlib...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "dlib"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ 使用 pip 安裝 dlib 成功")
            return True
        else:
            print(f"⚠️  pip 安裝失敗: {result.stderr}")
    except Exception as e:
        print(f"⚠️  pip 安裝出錯: {e}")
    
    return False

def main():
    print("🎯 簡化 dlib 安裝程序")
    print("=" * 40)
    
    if try_install_dlib():
        print("✅ dlib 安裝成功！")
        print("🎉 現在可以使用完整的臉部偵測功能")
    else:
        print("❌ 所有安裝方法都失敗了")
        print("💡 建議：")
        print("   1. 安裝 CMake: https://cmake.org/download/")
        print("   2. 安裝 Visual Studio Build Tools")
        print("   3. 或使用 OpenCV 模式（功能完整）")

if __name__ == "__main__":
    main() 