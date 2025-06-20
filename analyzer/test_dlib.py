#!/usr/bin/env python3
"""
測試 dlib 是否正常工作的腳本
"""

import sys
import os

def test_dlib():
    """測試 dlib 功能"""
    print("🧪 測試 dlib 功能...")
    print(f"Python 版本: {sys.version}")
    print(f"當前目錄: {os.getcwd()}")
    
    try:
        # 測試 dlib 導入
        print("正在導入 dlib...")
        import dlib
        print("✅ dlib 模組導入成功")
        print(f"dlib 版本: {dlib.__version__}")
        
        # 測試臉部偵測器
        print("正在初始化臉部偵測器...")
        detector = dlib.get_frontal_face_detector()
        print("✅ 臉部偵測器初始化成功")
        
        # 檢查模型檔案
        model_file = 'shape_predictor_68_face_landmarks.dat'
        print(f"檢查模型檔案: {model_file}")
        if os.path.exists(model_file):
            print(f"✅ 模型檔案存在: {model_file}")
            print(f"檔案大小: {os.path.getsize(model_file)} bytes")
            
            # 測試特徵點預測器
            print("正在初始化特徵點預測器...")
            predictor = dlib.shape_predictor(model_file)
            print("✅ 特徵點預測器初始化成功")
            
            print("\n🎉 dlib 所有功能測試通過！")
            return True
        else:
            print(f"❌ 模型檔案不存在: {model_file}")
            print("請執行 download_models.py 下載模型檔案")
            return False
            
    except ImportError as e:
        print(f"❌ dlib 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ dlib 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("開始測試...")
    result = test_dlib()
    print(f"測試結果: {'成功' if result else '失敗'}")
    input("按 Enter 鍵退出...") 