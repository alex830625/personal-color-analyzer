from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import colorsys
import dlib
import os
import uuid
import urllib.request
import bz2
import google.generativeai as genai
import json
import re
from bisenet import load_bisenet_model
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

# Ensure 'uploads' and 'debug_output' directories exist
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'debug_output')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '79999_iter.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 全域模型快取
_model_cache = {}
_model_lock = threading.Lock()

def get_bisenet_model():
    """取得 BiSeNet 模型（使用快取）"""
    with _model_lock:
        if 'bisenet' not in _model_cache:
            print("🔄 載入 BiSeNet 模型...")
            start_time = time.time()
            _model_cache['bisenet'] = load_bisenet_model(WEIGHT_PATH, device=DEVICE, n_classes=19)
            print(f"✅ BiSeNet 模型載入完成，耗時 {time.time() - start_time:.2f} 秒")
        return _model_cache['bisenet']

# 預載入模型
print("🚀 預載入 BiSeNet 模型...")
get_bisenet_model()

def download_dlib_model():
    """下載 dlib 臉部特徵點模型"""
    model_file = 'shape_predictor_68_face_landmarks.dat'
    if os.path.exists(model_file):
        return True
    
    print("📥 正在下載 dlib 模型檔案...")
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        # 下載壓縮檔案
        urllib.request.urlretrieve(model_url, compressed_file)
        print("✅ 模型檔案下載完成")
        
        # 解壓縮
        with bz2.open(compressed_file, 'rb') as source, open(model_file, 'wb') as target:
            target.write(source.read())
        print("✅ 模型檔案解壓縮完成")
        
        # 清理壓縮檔案
        os.remove(compressed_file)
        print("✅ 清理完成")
        return True
        
    except Exception as e:
        print(f"❌ 下載模型檔案失敗: {e}")
        return False

# 檢查並下載 dlib 模型
if not download_dlib_model():
    print("⚠️ 無法下載 dlib 模型，請手動下載 shape_predictor_68_face_landmarks.dat")
    print("下載網址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

# 初始化 dlib 的臉部偵測器和特徵點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
print("✅ dlib 模型載入成功")

def optimize_image_size(image, max_size=1024):
    """優化圖片尺寸以提升處理速度"""
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        return image
    
    # 計算縮放比例
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 使用 INTER_AREA 進行縮放（適合縮小）
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"📏 圖片已優化: {width}x{height} -> {new_width}x{new_height}")
    return resized

def rgb_to_hsv(rgb):
    """將RGB轉換為HSV"""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb(hsv):
    """將HSV轉換為RGB"""
    r, g, b = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    return (int(r*255), int(g*255), int(b*255))

def get_dominant_colors(image, k=1, mask=None):
    """
    獲取指定遮罩區域的「平均顏色」。
    """
    
    # 如果沒有遮罩，直接回傳預設顏色 (避免計算整張圖的平均色)
    if mask is None:
        return [(128, 128, 128)]
    
    # 使用 cv2.mean 計算遮罩區域的平均 BGR 值
    # mean() 回傳 (B, G, R, Alpha)
    mean_bgr = cv2.mean(image, mask=mask)
    
    # **關鍵修正**：將 BGR 轉換為 RGB
    # mean_bgr[0] 是 B, [1] 是 G, [2] 是 R
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

    # 將結果放入一個 list 中，以符合函式原有的回傳格式
    # 將顏色轉換為整數元組
    dominant_color = [tuple(map(int, mean_rgb))]
    
    return dominant_color

def detect_face_landmarks(image):
    """檢測臉部特徵點"""
    if predictor is None:
        return None, None
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 檢測臉部
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, None
    
    # 取第一個檢測到的臉部
    face = faces[0]
    
    # 獲取68個特徵點
    landmarks = predictor(gray, face)
    
    return face, landmarks

def get_landmark_points(landmarks):
    """將 dlib landmarks 轉換為 (x, y) 座標點的 list"""
    points = []
    for i in range(landmarks.num_parts):
        point = landmarks.part(i)
        points.append((point.x, point.y))
    return points

def extract_skin_region(image, landmarks):
    """
    針對 81 點模型優化臉型/下巴遮罩，若不足81點則fallback回68點遮罩。
    """
    if landmarks is None:
        return None
    points = get_landmark_points(landmarks)
    if len(points) >= 82:  # 81點模型
        jaw_points = [*range(0, 17), 79, 80, 81]
        jaw_pts = np.array([points[i] for i in jaw_points if i < len(points)], dtype=np.int32)
        face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        # 排除五官
        cv2.fillConvexPoly(face_mask, np.array(points[36:42], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[42:48], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[17:22], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[22:27], dtype=np.int32), 0)
        # 嘴唇（新：48-67+69-78）
        lip_points = [*range(48, 68), *range(69, 79)]
        lip_pts = np.array([points[i] for i in lip_points if i < len(points)], dtype=np.int32)
        cv2.fillConvexPoly(face_mask, lip_pts, 0)
        return face_mask
    else:  # fallback 68點
        face_outline_pts = np.array(points[0:17], dtype=np.int32)
        face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(face_outline_pts), 255)
        # 排除五官
        cv2.fillConvexPoly(face_mask, np.array(points[36:42], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[42:48], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[17:22], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[22:27], dtype=np.int32), 0)
        cv2.fillConvexPoly(face_mask, np.array(points[48:68], dtype=np.int32), 0)
        return face_mask

def extract_eye_regions(image, landmarks):
    """提取眼睛區域"""
    if landmarks is None:
        return []
    
    points = get_landmark_points(landmarks)
    eye_regions = []
    
    # 左眼 (36-41)
    left_eye_points = np.array(points[36:42], dtype=np.int32)
    # 右眼 (42-47)
    right_eye_points = np.array(points[42:48], dtype=np.int32)
    
    for eye_points in [left_eye_points, right_eye_points]:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 使用凸包更精準框選眼睛
        hull = cv2.convexHull(eye_points)
        cv2.fillConvexPoly(mask, hull, 255)
        eye_regions.append(mask)
    
    return eye_regions

def extract_lip_region(image, landmarks):
    """
    針對 81 點模型優化嘴唇遮罩，若不足81點則fallback回68點遮罩。
    """
    if landmarks is None:
        return None
    points = get_landmark_points(landmarks)
    if len(points) >= 79:  # 81點模型
        lip_points = [*range(48, 68), *range(69, 79)]
        lip_pts = np.array([points[i] for i in lip_points if i < len(points)], dtype=np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, lip_pts, 255)
        return mask
    else:  # fallback 68點
        lip_points = np.array(points[48:68], dtype=np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, lip_points, 255)
        return mask

def extract_hair_mask_bisenet(image):
    """使用 BiSeNet 提取頭髮遮罩"""
    try:
        model = get_bisenet_model()
        
        # 預處理圖片
        img = cv2.resize(image, (512, 512))
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        
        # 推理
        with torch.no_grad():
            out, _, _ = model(img)
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
        
        # 頭髮類別為 17
        hair_mask = (pred == 17).astype(np.uint8) * 255
        
        # 調整回原始尺寸
        hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return hair_mask
    except Exception as e:
        print(f"⚠️ BiSeNet 頭髮提取失敗: {e}")
        return None

def extract_hair_region(image, landmarks, face_rect):
    """使用傳統方法提取頭髮區域（fallback）"""
    if landmarks is None or face_rect is None:
        return None
    
    points = get_landmark_points(landmarks)
    
    # 定義頭髮區域的關鍵點
    hair_points = []
    
    # 額頭區域 (68點模型: 17-21, 22-26)
    if len(points) >= 27:
        hair_points.extend(points[17:22])  # 左眉毛
        hair_points.extend(points[22:27])  # 右眉毛
    
    # 頭頂區域（估計）
    if len(points) >= 27:
        # 使用眉毛中點向上延伸
        left_brow_center = points[19]
        right_brow_center = points[24]
        brow_center = ((left_brow_center[0] + right_brow_center[0]) // 2,
                      (left_brow_center[1] + right_brow_center[1]) // 2)
        
        # 向上延伸約 1.5 倍臉部高度
        face_height = face_rect.height()
        hair_top = (brow_center[0], max(0, brow_center[1] - int(face_height * 1.5)))
        
        hair_points.append(hair_top)
    
    if len(hair_points) < 3:
        return None
    
    # 創建頭髮遮罩
    hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hair_pts = np.array(hair_points, dtype=np.int32)
    cv2.fillConvexPoly(hair_mask, cv2.convexHull(hair_pts), 255)
    
    return hair_mask

def gray_world_white_balance(img):
    # 灰世界假設自動白平衡
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def mask_skin_pixels(image, mask):
    # YCrCb 膚色像素過濾
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    return cv2.bitwise_and(mask, skin_mask)

def calculate_skin_tone(image, landmarks):
    """計算膚色（優化版本）"""
    skin_mask = extract_skin_region(image, landmarks)
    if skin_mask is None:
        return (128, 128, 128)
    
    # 應用膚色像素過濾
    filtered_mask = mask_skin_pixels(image, skin_mask)
    
    # 計算平均膚色
    skin_colors = get_dominant_colors(image, mask=filtered_mask)
    return skin_colors[0] if skin_colors else (128, 128, 128)

def create_debug_image_dlib(image, face_rect, skin_mask, eye_masks, lip_mask, hair_mask, output_path):
    """創建調試圖片（優化版本）"""
    debug_img = image.copy()
    
    # 定義顏色
    SKIN_COLOR = (0, 255, 0)    # 綠色
    EYE_COLOR = (255, 0, 0)     # 藍色
    LIP_COLOR = (0, 0, 255)     # 紅色
    HAIR_COLOR = (255, 255, 0)  # 青色
    
    # 繪製遮罩
    if skin_mask is not None:
        skin_overlay = np.zeros_like(debug_img)
        skin_overlay[skin_mask == 255] = SKIN_COLOR
        cv2.addWeighted(debug_img, 1, skin_overlay, 0.3, 0, debug_img)
    
    if eye_masks:
        for eye_mask in eye_masks:
            eye_overlay = np.zeros_like(debug_img)
            eye_overlay[eye_mask == 255] = EYE_COLOR
            cv2.addWeighted(debug_img, 1, eye_overlay, 0.5, 0, debug_img)
    
    if lip_mask is not None:
        lip_overlay = np.zeros_like(debug_img)
        lip_overlay[lip_mask == 255] = LIP_COLOR
        cv2.addWeighted(debug_img, 1, lip_overlay, 0.5, 0, debug_img)
    
    if hair_mask is not None:
        hair_overlay = np.zeros_like(debug_img)
        hair_overlay[hair_mask == 255] = HAIR_COLOR
        cv2.addWeighted(debug_img, 1, hair_overlay, 0.4, 0, debug_img)
    
    cv2.imwrite(output_path, debug_img)

def create_debug_mask_image(image, skin_mask=None, eye_masks=None, lip_mask=None, hair_mask=None, output_path='debug_mask.png'):
    """創建調試遮罩圖片（優化版本）"""
    debug_img = np.zeros_like(image)
    
    # 定義顏色
    SKIN_COLOR = (0, 255, 0)    # 綠色
    EYE_COLOR = (255, 0, 0)     # 藍色
    LIP_COLOR = (0, 0, 255)     # 紅色
    HAIR_COLOR = (255, 255, 0)  # 青色
    
    # 繪製遮罩
    if skin_mask is not None:
        skin_overlay = np.zeros_like(debug_img)
        skin_overlay[skin_mask == 255] = SKIN_COLOR
        cv2.addWeighted(debug_img, 1, skin_overlay, 0.5, 0, debug_img)
    
    if eye_masks:
        for eye_mask in eye_masks:
            eye_overlay = np.zeros_like(debug_img)
            eye_overlay[eye_mask == 255] = EYE_COLOR
            cv2.addWeighted(debug_img, 1, eye_overlay, 0.5, 0, debug_img)
    
    if lip_mask is not None:
        lip_overlay = np.zeros_like(debug_img)
        lip_overlay[lip_mask == 255] = LIP_COLOR
        cv2.addWeighted(debug_img, 1, lip_overlay, 0.5, 0, debug_img)
    
    if hair_mask is not None:
        hair_overlay = np.zeros_like(debug_img)
        hair_overlay[hair_mask == 255] = HAIR_COLOR
        cv2.addWeighted(debug_img, 1, hair_overlay, 0.4, 0, debug_img)
    
    cv2.imwrite(output_path, debug_img)

def analyze_seasonal_colors(skin_tone, eye_color, hair_color):
    """分析季節性色彩"""
    # 轉換為HSV進行分析
    skin_hsv = rgb_to_hsv(skin_tone)
    eye_hsv = rgb_to_hsv(eye_color)
    hair_hsv = rgb_to_hsv(hair_color)
    
    # 分析膚色的色調和飽和度
    skin_hue = skin_hsv[0] * 360  # 轉換為度數
    skin_sat = skin_hsv[1]
    skin_val = skin_hsv[2]
    
    # 季節性色彩判斷邏輯
    if skin_hue < 30 or skin_hue > 330:  # 偏紅/橙
        if skin_sat > 0.3:  # 飽和度較高
            season = "spring"
        else:
            season = "autumn"
    elif 30 <= skin_hue <= 90:  # 偏黃
        if skin_val > 0.6:  # 亮度較高
            season = "spring"
        else:
            season = "autumn"
    elif 90 <= skin_hue <= 150:  # 偏綠
        season = "summer"
    elif 150 <= skin_hue <= 270:  # 偏藍/紫
        season = "winter"
    else:  # 270-330 偏紫/紅
        season = "winter"
    
    return season, {
        'skin_hsv': skin_hsv,
        'eye_hsv': eye_hsv,
        'hair_hsv': hair_hsv
    }

def get_color_palette(season):
    """根據季節獲取建議色板"""
    palettes = {
        "spring": {
            "clothes": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            "makeup": ["#FFB6C1", "#FFC0CB", "#FF69B4", "#FF1493", "#DC143C"],
            "jewelry": ["#FFD700", "#FFA500", "#FF6347", "#FF4500"],
            "avoid": ["#000080", "#191970", "#483D8B", "#6A5ACD"]
        },
        "summer": {
            "clothes": ["#87CEEB", "#98FB98", "#DDA0DD", "#F0E68C", "#FFB6C1"],
            "makeup": ["#E6E6FA", "#D8BFD8", "#DDA0DD", "#EE82EE", "#DA70D6"],
            "jewelry": ["#C0C0C0", "#E6E6FA", "#F0F8FF", "#F5F5DC"],
            "avoid": ["#FF4500", "#FF6347", "#FF8C00", "#FFA500"]
        },
        "autumn": {
            "clothes": ["#8B4513", "#A0522D", "#CD853F", "#D2691E", "#B8860B"],
            "makeup": ["#DEB887", "#F4A460", "#DAA520", "#B8860B", "#CD853F"],
            "jewelry": ["#B8860B", "#DAA520", "#CD853F", "#8B4513"],
            "avoid": ["#00CED1", "#40E0D0", "#48D1CC", "#20B2AA"]
        },
        "winter": {
            "clothes": ["#000080", "#191970", "#483D8B", "#6A5ACD", "#9370DB"],
            "makeup": ["#FF1493", "#DC143C", "#B22222", "#8B0000", "#FF4500"],
            "jewelry": ["#C0C0C0", "#FFFFFF", "#000000", "#4169E1"],
            "avoid": ["#FFD700", "#FFA500", "#FF6347", "#FF4500"]
        }
    }
    
    return palettes.get(season, palettes["spring"])

def rgb_to_hex(rgb):
    """將RGB轉換為十六進制"""
    return '#%02x%02x%02x' % rgb

def get_color_names_from_gemini(hex_colors):
    """從 Gemini 獲取顏色名稱（優化版本）"""
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        prompt = (
            "請幫我判斷以下色碼的繁體中文名稱，回傳格式為 JSON 物件，key 為色碼，value 為中文名稱：\n"
            + ", ".join(hex_colors)
        )
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        # 解析 Gemini 回傳的 JSON 內容
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {c: "未知色名" for c in hex_colors}
    except Exception as e:
        print(f"⚠️ Gemini 色名查詢失敗: {e}")
        return {c: "未知色名" for c in hex_colors}

def process_image_parallel(image, landmarks, face):
    """並行處理圖片分析"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交並行任務
        skin_future = executor.submit(calculate_skin_tone, image, landmarks)
        eye_future = executor.submit(extract_eye_regions, image, landmarks)
        lip_future = executor.submit(extract_lip_region, image, landmarks)
        hair_future = executor.submit(extract_hair_mask_bisenet, image)
        
        # 等待結果
        skin_tone = skin_future.result()
        eye_masks = eye_future.result()
        lip_mask = lip_future.result()
        hair_mask = hair_future.result()
        
        # 處理眼睛顏色
        all_eye_colors = []
        if eye_masks:
            for eye_mask in eye_masks:
                all_eye_colors.extend(get_dominant_colors(image, mask=eye_mask))
        eye_color = all_eye_colors[0] if all_eye_colors else (128, 128, 128)
        
        # 處理其他顏色
        lip_color = get_dominant_colors(image, mask=lip_mask)[0] if lip_mask is not None else (128, 128, 128)
        hair_color = get_dominant_colors(image, mask=hair_mask)[0] if hair_mask is not None else (128, 128, 128)
        
        return skin_tone, eye_color, lip_color, hair_color, eye_masks, lip_mask, hair_mask

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({"error": "請求中缺少圖片檔案"}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400

    start_time = time.time()
    print(f"🚀 開始分析圖片: {file.filename}")

    # Save uploaded file
    filename = f"upload_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    file.save(file_path)

    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "無法讀取圖片檔案"}), 500

    try:
        # 優化圖片尺寸
        image = optimize_image_size(image, max_size=1024)
        
        # --- 1. Face and Landmark Detection ---
        print("🔍 檢測臉部特徵點...")
        face, landmarks = detect_face_landmarks(image)
        if landmarks is None:
            return jsonify({"error": "未偵測到臉部特徵點，請使用更清晰的正面照。"}), 400

        # --- 2. 並行分析顏色 ---
        print("🎨 分析顏色...")
        skin_tone, eye_color, lip_color, hair_color, eye_masks, lip_mask, hair_mask = process_image_parallel(image, landmarks, face)

        # --- 3. Gemini 查詢色名（可選，如果失敗不影響主要分析） ---
        print("🌈 查詢顏色名稱...")
        hex_colors = [
            rgb_to_hex(skin_tone),
            rgb_to_hex(eye_color),
            rgb_to_hex(hair_color),
            rgb_to_hex(lip_color)
        ]
        color_names = get_color_names_from_gemini(hex_colors)

        # --- 4. Seasonal Analysis ---
        print("🌤️ 分析季節性色彩...")
        season, analysis_details = analyze_seasonal_colors(skin_tone, eye_color, hair_color)

        # --- 5. 可選：創建 Debug 圖片（僅在需要時） ---
        debug_image_url = ""
        debug_mask_url = ""
        if os.environ.get('DEBUG_MODE', 'false').lower() == 'true':
            print("📸 創建調試圖片...")
            skin_mask_for_debug = extract_skin_region(image, landmarks)
            debug_image_filename = f"debug_{uuid.uuid4().hex}.jpg"
            debug_image_path = os.path.join(DEBUG_OUTPUT_DIR, debug_image_filename)
            create_debug_image_dlib(image, face, skin_mask_for_debug, eye_masks, lip_mask, hair_mask, debug_image_path)
            debug_image_url = f"/debug/{debug_image_filename}"

            debug_mask_filename = f"debug_mask_{uuid.uuid4().hex}.png"
            debug_mask_path = os.path.join(DEBUG_OUTPUT_DIR, debug_mask_filename)
            create_debug_mask_image(image, skin_mask_for_debug, eye_masks, lip_mask, hair_mask, debug_mask_path)
            debug_mask_url = f"/debug/{debug_mask_filename}"

        # --- 6. Prepare Response ---
        response_data = {
            "skin_tone": rgb_to_hex(skin_tone),
            "eye_color": rgb_to_hex(eye_color),
            "hair_color": rgb_to_hex(hair_color),
            "lip_color": rgb_to_hex(lip_color),
            "color_names": color_names,
            "season": season,
            "season_name": {"spring": "春季型", "summer": "夏季型", "autumn": "秋季型", "winter": "冬季型"}.get(season, "分析失敗"),
            "color_suggestions": get_color_palette(season),
            "analysis_details": analysis_details,
            "debug_image_url": debug_image_url,
            "debug_mask_url": debug_mask_url,
            "debug_error": "",
            "processing_time": round(time.time() - start_time, 2)
        }
        
        print(f"✅ 分析完成，耗時 {response_data['processing_time']} 秒")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"❌ 分析失敗: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"伺服器在進階分析中發生嚴重錯誤: {e}"}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/debug/<filename>')
def serve_debug_image(filename):
    return send_from_directory(DEBUG_OUTPUT_DIR, filename)

if __name__ == '__main__':
    # ... (add predictor file check)
    app.run(host='0.0.0.0', port=5001, threaded=True)