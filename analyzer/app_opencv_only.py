from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import colorsys
import os
import uuid
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

# 創建並配置偵錯圖片資料夾
DEBUG_FOLDER = 'debug_output'
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)
app.config['DEBUG_FOLDER'] = DEBUG_FOLDER

# Ensure 'uploads' and 'debug_output' directories exist
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'debug_output')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

def bgr_to_hex(bgr):
    return '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (4, 2, 0)], dtype=np.uint8)

def rgb_to_hsv(rgb):
    # Note: input is BGR, but Python's colorsys expects RGB, so we use indices 2,1,0
    r, g, b = rgb[2]/255.0, rgb[1]/255.0, rgb[0]/255.0
    return colorsys.rgb_to_hsv(r, g, b)

def get_dominant_colors(image, k=5, mask=None):
    """獲取主要顏色"""
    if mask is not None:
        # 使用遮罩只分析特定區域
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        img = masked_img.reshape((-1, 3))
    else:
        img = image.reshape((-1, 3))
    
    # 過濾黑色和白色像素
    img = img[np.all(img > 20, axis=1)]  # 過濾太暗的像素
    img = img[np.any(img < 235, axis=1)]  # 過濾太亮的像素
    
    if len(img) == 0:
        return [(128, 128, 128)]  # 如果沒有有效像素，返回灰色
    
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 計算每個聚類的像素數量
    counts = np.bincount(labels.flatten())
    # 按像素數量排序
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = [tuple(map(int, centers[i])) for i in sorted_indices]
    
    return dominant_colors

def detect_face_opencv(image):
    """使用OpenCV檢測臉部"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用OpenCV的Haar級聯分類器檢測臉部
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None
    
    # 返回第一個檢測到的臉部
    x, y, w, h = faces[0]
    return (x, y, w, h)

def extract_skin_region_opencv(image, face_rect):
    """更精確地提取膚色區域（臉部中央、兩頰、額頭）"""
    if face_rect is None:
        return None
    x, y, w, h = face_rect
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # 臉部中央
    center = (x + int(w*0.35), y + int(h*0.35), int(w*0.3), int(h*0.3))
    # 左頰
    cheek_left = (x + int(w*0.13), y + int(h*0.45), int(w*0.18), int(h*0.18))
    # 右頰
    cheek_right = (x + int(w*0.69), y + int(h*0.45), int(w*0.18), int(h*0.18))
    # 額頭
    forehead = (x + int(w*0.28), y + int(h*0.13), int(w*0.44), int(h*0.18))
    for region in [center, cheek_left, cheek_right, forehead]:
        cv2.rectangle(mask, (region[0], region[1]), (region[0]+region[2], region[1]+region[3]), 255, -1)
    return mask

def get_skin_pixels(image, mask):
    """只保留 YCrCb 膚色像素"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    if mask is not None:
        skin_mask = cv2.bitwise_and(skin_mask, mask)
    skin_pixels = image[skin_mask > 0]
    return skin_pixels

def detect_skin_tone_opencv(image, face_rect, skin_mask):
    """優化後的膚色偵測"""
    if face_rect is None:
        raise ValueError('未偵測到臉部，請上傳清晰正面照')
    if skin_mask is None:
        raise ValueError('膚色區域提取失敗')
    skin_pixels = get_skin_pixels(image, skin_mask)
    if len(skin_pixels) < 10:
        raise ValueError('膚色像素過少，請換一張照片')
    # K-means 聚類
    img = np.float32(skin_pixels)
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant = tuple(map(int, centers[np.argmax(counts)]))
    return dominant

def extract_eye_regions_opencv(image, face_rect):
    """使用OpenCV提取眼睛區域"""
    if face_rect is None:
        return []
    
    x, y, w, h = face_rect
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用Haar級聯分類器檢測眼睛
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # 在臉部區域內檢測眼睛
    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)
    
    eye_regions = []
    for (ex, ey, ew, eh) in eyes:
        # 創建眼睛遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 調整眼睛座標到原圖
        eye_x = x + ex
        eye_y = y + ey
        
        # 擴展眼睛區域
        expanded_rect = (eye_x-5, eye_y-5, ew+10, eh+10)
        cv2.rectangle(mask, (expanded_rect[0], expanded_rect[1]), 
                     (expanded_rect[0] + expanded_rect[2], expanded_rect[1] + expanded_rect[3]), 255, -1)
        
        eye_regions.append(mask)
    
    return eye_regions

def extract_hair_region_opencv(image, face_rect):
    """使用OpenCV提取頭髮區域"""
    if face_rect is None:
        return None
    
    x, y, w, h = face_rect
    
    # 創建遮罩
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 頭髮區域通常在臉部上方和側面
    # 上方頭髮區域
    hair_top = (x - int(w*0.2), y - int(h*0.8), int(w*1.4), int(h*0.8))
    
    # 側面頭髮區域
    hair_left = (x - int(w*0.3), y - int(h*0.5), int(w*0.3), int(h*1.2))
    hair_right = (x + int(w*1.0), y - int(h*0.5), int(w*0.3), int(h*1.2))
    
    # 繪製頭髮區域
    cv2.rectangle(mask, (hair_top[0], hair_top[1]), 
                 (hair_top[0] + hair_top[2], hair_top[1] + hair_top[3]), 255, -1)
    cv2.rectangle(mask, (hair_left[0], hair_left[1]), 
                 (hair_left[0] + hair_left[2], hair_left[1] + hair_left[3]), 255, -1)
    cv2.rectangle(mask, (hair_right[0], hair_right[1]), 
                 (hair_right[0] + hair_right[2], hair_right[1] + hair_right[3]), 255, -1)
    
    return mask

def get_dark_pixels(image, mask, threshold=120):
    """只保留深色像素（排除高亮反光）"""
    if mask is not None:
        region = image[mask > 0]
    else:
        region = image.reshape(-1, 3)
    # 只保留亮度較低的像素
    hsv = cv2.cvtColor(region.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    dark_pixels = region[hsv[:,2] < threshold]
    return dark_pixels

def detect_eye_color_opencv(image, face_rect, eye_regions):
    """優化後的眼睛顏色偵測"""
    if face_rect is None:
        raise ValueError('未偵測到臉部，請上傳清晰正面照')
    if not eye_regions:
        raise ValueError('未偵測到眼睛，請換一張照片')
    # 只取最大兩個區域
    if len(eye_regions) > 2:
        eye_regions = sorted(eye_regions, key=lambda m: np.sum(m>0), reverse=True)[:2]
    all_eye_pixels = np.vstack([get_dark_pixels(image, mask, threshold=110) for mask in eye_regions if np.sum(mask)>0])
    if len(all_eye_pixels) < 10:
        raise ValueError('眼睛像素過少，請換一張照片')
    img = np.float32(all_eye_pixels)
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant = tuple(map(int, centers[np.argmax(counts)]))
    return dominant

def detect_hair_color_opencv(image, face_rect, hair_mask):
    """優化後的頭髮顏色偵測"""
    if face_rect is None:
        raise ValueError('未偵測到臉部，請上傳清晰正面照')
    if hair_mask is None:
        raise ValueError('頭髮區域提取失敗')
    hair_pixels = get_dark_pixels(image, hair_mask, threshold=100)
    if len(hair_pixels) < 10:
        raise ValueError('頭髮像素過少，請換一張照片')
    img = np.float32(hair_pixels)
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant = tuple(map(int, centers[np.argmax(counts)]))
    return dominant

def detect_skin_tone_fallback(image):
    """傳統膚色檢測方法（備用）"""
    # 轉換到YCrCb色彩空間，更適合膚色檢測
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # 膚色範圍（YCrCb空間）
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # 創建膚色遮罩
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 形態學操作改善遮罩
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # 獲取膚色區域的主要顏色
    skin_colors = get_dominant_colors(image, k=3, mask=skin_mask)
    
    if skin_colors:
        return skin_colors[0]  # 返回最主要的膚色
    else:
        # 如果沒有檢測到膚色，使用整體主色
        return get_dominant_colors(image, k=3)[0]

def detect_eye_color_fallback(image):
    """傳統眼睛顏色檢測方法（備用）"""
    # 轉換到灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用Haar級聯分類器檢測眼睛
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    
    eye_colors = []
    for (x, y, w, h) in eyes:
        # 提取眼睛區域
        eye_roi = image[y:y+h, x:x+w]
        if eye_roi.size > 0:
            # 獲取眼睛區域的主要顏色
            colors = get_dominant_colors(eye_roi, k=3)
            eye_colors.extend(colors)
    
    if eye_colors:
        return eye_colors[0]  # 返回最主要的眼睛顏色
    else:
        # 如果沒有檢測到眼睛，使用整體主色
        return get_dominant_colors(image, k=5)[0]

def detect_hair_color_fallback(image):
    """傳統頭髮顏色檢測方法（備用）"""
    # 轉換到HSV色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 頭髮顏色範圍（深色系）
    lower_hair = np.array([0, 0, 0], dtype=np.uint8)
    upper_hair = np.array([180, 255, 100], dtype=np.uint8)
    
    # 創建頭髮遮罩
    hair_mask = cv2.inRange(hsv, lower_hair, upper_hair)
    
    # 形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    
    # 獲取頭髮區域的主要顏色
    hair_colors = get_dominant_colors(image, k=3, mask=hair_mask)
    
    if hair_colors:
        return hair_colors[0]  # 返回最主要的頭髮顏色
    else:
        # 如果沒有檢測到頭髮，使用整體主色
        return get_dominant_colors(image, k=7)[0]

def analyze_seasonal_colors(skin_bgr, eye_bgr, hair_bgr):
    skin_hsv = rgb_to_hsv(skin_bgr)
    eye_hsv = rgb_to_hsv(eye_bgr)
    hair_hsv = rgb_to_hsv(hair_bgr)
    
    skin_hue, skin_sat, skin_val = skin_hsv[0] * 360, skin_hsv[1], skin_hsv[2]
    eye_sat = eye_hsv[1]
    hair_sat = hair_hsv[1]
    
    warm_cool_score = 0
    if 0 <= skin_hue <= 50 or 330 <= skin_hue <= 360: warm_cool_score += 1
    else: warm_cool_score -= 1
    if hair_sat > 0.3: warm_cool_score += 1
    else: warm_cool_score -= 1
        
    if warm_cool_score > 0: # Warm Tones: Spring or Autumn
        if skin_sat > 0.4 and eye_sat > 0.35: season = "spring"
        else: season = "autumn"
    else: # Cool Tones: Summer or Winter
        if skin_val > 0.75 and hair_sat < 0.35: season = "summer"
        else: season = "winter"
            
    analysis_details = { "skin_hsv": skin_hsv, "eye_hsv": eye_hsv, "hair_hsv": hair_hsv }
    return season, analysis_details

def get_color_palette(season):
    palettes = {
        "spring": {"clothes": ["#FFDAB9", "#90EE90"], "makeup": ["#FFB6C1", "#FFA07A"]},
        "summer": {"clothes": ["#ADD8E6", "#E6E6FA"], "makeup": ["#DDA0DD", "#DB7093"]},
        "autumn": {"clothes": ["#F4A460", "#8B4513"], "makeup": ["#CD853F", "#B22222"]},
        "winter": {"clothes": ["#000080", "#FFFFFF"], "makeup": ["#FF0000", "#191970"]},
    }
    return palettes.get(season, palettes["winter"])

def get_dominant_color(roi, k=3):
    if roi is None or roi.size == 0: return None
    try:
        pixels = roi.reshape(-1, 3)
        clt = KMeans(n_clusters=k, n_init=3, random_state=42)
        clt.fit(pixels)
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=num_labels)
        hist = hist.astype("float")
        hist /= hist.sum()
        dominant_color = clt.cluster_centers_[np.argmax(hist)]
        return bgr_to_hex(dominant_color.astype(int))
    except Exception as e:
        print(f"[*] K-Means failed: {e}. Falling back to average.")
        avg_color = np.mean(roi, axis=(0, 1)).astype(int)
        return bgr_to_hex(avg_color)

def create_debug_image(image, face_rect, skin_rect, eye_rects, hair_rect, output_path):
    print(f"  [DEBUG] -> 正在建立偵錯圖，將儲存至: {output_path}")
    debug_img = image.copy()
    try:
        # Draw face rect (Green)
        if face_rect:
            x, y, w, h = face_rect
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw skin rect (Red)
        if skin_rect:
            x, y, w, h = skin_rect
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Draw eye rects (Blue)
        if eye_rects:
            for (ex, ey, ew, eh) in eye_rects:
                cv2.rectangle(debug_img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        # Draw hair rect (Yellow)
        if hair_rect:
            x, y, w, h = hair_rect
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        cv2.imwrite(output_path, debug_img)
        print(f"  [SUCCESS] -> 偵錯圖已成功儲存。")
    except Exception as e:
        print(f"  [ERROR] -> 建立偵錯圖失敗: {e}")
        # Even if it fails, we continue, the URL will just lead to a 404
        # but the main analysis result can still be returned.

def analyze_image(image_path):
    print("\n" + "="*20 + " 開始新分析 (OpenCV) " + "="*20)
    print(f"1. 讀取圖片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("   [ERROR] cv2.imread 無法讀取圖片。")
        return {"error": "無法讀取圖片檔案。"}

    # --- 1. Face Detection ---
    print("2. 進行臉部偵測...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print("   [FAILED] 未偵測到臉部。")
        return {"error": "未偵測到臉部，請嘗試使用更清晰、臉部更突出的照片。"}
    
    print(f"   [SUCCESS] 偵測到 {len(faces)} 個臉部，使用第一個。")
    x, y, w, h = faces[0]
    face_rect = (x, y, w, h)

    # Initialize results and rects
    skin_tone_hex, eye_color_hex, hair_color_hex = None, None, None
    skin_rect, hair_rect = None, None
    eye_rects = []

    # --- 2. Skin Analysis ---
    try:
        skin_x, skin_y = x + w // 4, y + h // 2
        skin_w, skin_h = w // 2, h // 4
        skin_rect = (skin_x, skin_y, skin_w, skin_h)
        skin_roi = image[skin_y:skin_y+skin_h, skin_x:skin_x+skin_w]
        if skin_roi.size > 0:
            skin_tone_hex = get_dominant_color(skin_roi)
        else: raise ValueError("Skin ROI is empty")
    except Exception as e:
        print(f"[*] Skin analysis failed: {e}")

    # --- 3. Eye Analysis ---
    try:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        
        if len(eyes) > 0:
            eye_rois = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_full_x, eye_full_y = x + ex, y + ey
                eye_rects.append((eye_full_x, eye_full_y, ew, eh))
                eye_rois.append(image[eye_full_y:eye_full_y+eh, eye_full_x:eye_full_x+ew])
            
            # THE FIX: Instead of trying to combine ROIs, just use the first one.
            # This is robust and prevents crashes from different ROI sizes.
            if eye_rois:
                print(f"   [INFO] 偵測到 {len(eye_rois)} 個眼睛區域，使用第一個進行分析。")
                eye_color_hex = get_dominant_color(eye_rois[0])
        else:
            print("   [INFO] 在臉部區域未偵測到眼睛。")

    except Exception as e: 
        print(f"[*] 眼睛分析失敗: {e}")

    # --- 4. Hair Analysis ---
    try:
        hair_x, hair_y = x + w // 4, y
        hair_w, hair_h = w // 2, h // 4
        hair_rect = (hair_x, hair_y, hair_w, hair_h)
        hair_roi = image[hair_y:hair_y+hair_h, hair_x:hair_x+hair_w]
        if hair_roi.size > 0:
            hair_color_hex = get_dominant_color(hair_roi, k=5)
        else: raise ValueError("Hair ROI is empty")
    except Exception as e:
        print(f"[*] Hair analysis failed: {e}")

    # --- 4. Seasonal Analysis ---
    print("4. 進行季節性色彩分析...")
    season, analysis_details = "N/A", {}
    if skin_tone_hex and eye_color_hex and hair_color_hex:
        try:
            skin_bgr = hex_to_bgr(skin_tone_hex)
            eye_bgr = hex_to_bgr(eye_color_hex)
            hair_bgr = hex_to_bgr(hair_color_hex)
            season, analysis_details = analyze_seasonal_colors(skin_bgr, eye_bgr, hair_bgr)
            print(f"   [SUCCESS] 季節分析結果: {season}")
        except Exception as e:
            print(f"   [ERROR] 季節分析失敗: {e}")
    else:
        print("   [SKIPPED] 因缺少顏色資訊，跳過季節分析。")

    # --- 5. Create Debug Image ---
    print("5. 建立視覺化偵錯圖...")
    debug_image_filename = f"debug_{uuid.uuid4().hex}.jpg"
    debug_image_path = os.path.join(DEBUG_OUTPUT_DIR, debug_image_filename)
    create_debug_image(image, face_rect, skin_rect, eye_rects, hair_rect, debug_image_path)
    # This variable holds the URL that MUST be in the final response.
    debug_image_url = f"/debug/{debug_image_filename}"

    # --- 6. Prepare Final Response ---
    print("6. 準備最終回應資料...")
    
    # Construct the response dictionary piece by piece to ensure correctness
    response_data = {
        "skin_tone": skin_tone_hex,
        "eye_color": eye_color_hex,
        "hair_color": hair_color_hex,
        "season": season,
        "season_name": {"spring": "春季型", "summer": "夏季型", "autumn": "秋季型", "winter": "冬季型"}.get(season, "分析失敗"),
        "color_suggestions": get_color_palette(season) if season != "N/A" else {},
        "analysis_details": analysis_details,
        # THE CRITICAL FIX: Explicitly adding the debug image URL and error fields
        "debug_image_url": debug_image_url,
        "debug_error": ""
    }
    
    failed_parts = []
    if skin_tone_hex is None: failed_parts.append("皮膚")
    if eye_color_hex is None: failed_parts.append("眼睛")
    if hair_color_hex is None: failed_parts.append("頭髮")
    if failed_parts:
        response_data["debug_error"] = f"無法精準分析以下區域: {', '.join(failed_parts)}。偵錯圖可能不完整。"

    print(f"   [INFO] 回應資料: {response_data}")
    print("="*20 + " 分析結束 " + "="*20)
    return response_data

@app.route('/analyze', methods=['POST'])
def handle_analyze():
    if 'photo' not in request.files:
        return jsonify({"error": "請求中缺少圖片檔案"}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400

    filename = f"upload_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    file.save(file_path)

    try:
        result = analyze_image(file_path)
        
        # THE ULTIMATE DEBUG STEP: Log the dictionary right before it's sent.
        print("\n" + "#"*15 + " FINAL SERVER RESPONSE " + "#"*15)
        print("準備透過 jsonify 回傳的最終字典:")
        print(result)
        print("#"*55 + "\n")

        # Create a response object and add anti-caching headers
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response

    except Exception as e:
        print(f"[!] Critical error in analyze route: {e}")
        return jsonify({"error": "伺服器內部發生嚴重錯誤。"}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/debug/<filename>')
def serve_debug_image(filename):
    return send_from_directory(DEBUG_OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 