#!/usr/bin/env python3
"""
下載 dlib 臉部特徵點模型
"""

import urllib.request
import os
import sys

def download_file(url, filename):
    """下載文件"""
    print(f"正在下載 {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ {filename} 下載完成！")
        return True
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False

def main():
    # 模型文件 URL
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_filename = "shape_predictor_68_face_landmarks.dat.bz2"
    
    # 檢查文件是否已存在
    if os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("✅ 模型文件已存在，跳過下載")
        return
    
    # 下載模型文件
    if download_file(model_url, model_filename):
        # 解壓縮文件
        print("正在解壓縮...")
        import bz2
        try:
            with bz2.open(model_filename, 'rb') as source, open('shape_predictor_68_face_landmarks.dat', 'wb') as target:
                target.write(source.read())
            print("✅ 解壓縮完成！")
            
            # 刪除壓縮文件
            os.remove(model_filename)
            print("✅ 清理完成！")
        except Exception as e:
            print(f"❌ 解壓縮失敗: {e}")
            sys.exit(1)
    else:
        print("❌ 無法下載模型文件")
        sys.exit(1)

if __name__ == "__main__":
    main() 