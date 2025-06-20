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

app = Flask(__name__)
CORS(app)

# Ensure 'uploads' and 'debug_output' directories exist
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'debug_output')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '79999_iter.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
bisenet_model = load_bisenet_model(WEIGHT_PATH, device=DEVICE, n_classes=19)

print("模型權重路徑：", WEIGHT_PATH)
print("檔案是否存在：", os.path.exists(WEIGHT_PATH))

def download_dlib_model():
    """下載 dlib 臉部特徵點模型"""
    model_file = 'shape_predictor_68_face_landmarks.dat'
    if os.path.exists(model_file):
        print("✅ dlib 模型檔案已存在")
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
        print("❌ dlib 模型未載入，無法進行臉部特徵點檢測")
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
        hull = cv2.convexHull(lip_pts)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    else:  # fallback 68點
        lip_points = np.array(points[48:68], dtype=np.int32)
        hull = cv2.convexHull(lip_points)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

def extract_hair_mask_bisenet(image):
    import cv2
    import torch
    h, w = image.shape[:2]
    # BiSeNet 輸入需 512x512
    img_resized = cv2.resize(image, (512, 512))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) / 255.0
    with torch.no_grad():
        out = bisenet_model(img_tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    # 取出頭髮區域
    hair_mask = (parsing == 17).astype(np.uint8) * 255
    # 還原到原圖大小
    hair_mask_full = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return hair_mask_full

def gray_world_white_balance(img):
    # 灰世界假設自動白平衡
    img = img.astype(np.float32)
    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    img[:,:,0] *= scale_b
    img[:,:,1] *= scale_g
    img[:,:,2] *= scale_r
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def mask_skin_pixels(image, mask):
    # YCrCb 膚色像素過濾
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    combined = cv2.bitwise_and(mask, skin_mask)
    # Debug: 印出膚色像素數量與平均 BGR
    skin_pixels = image[combined > 0]
    print(f"[膚色過濾 debug] 膚色像素數量: {len(skin_pixels)}")
    if len(skin_pixels) > 0:
        mean_bgr = np.mean(skin_pixels, axis=0)
        print(f"[膚色過濾 debug] 膚色像素平均 BGR: {mean_bgr}")
    else:
        print("[膚色過濾 debug] 無膚色像素")
    return combined

def calculate_skin_tone(image, landmarks):
    """
    一次性優化：自動白平衡、LAB色空間、五區分區、膚色像素過濾、加權平均
    """
    if landmarks is None:
        return (128, 128, 128)
    # 1. 自動白平衡
    image = gray_world_white_balance(image)
    points = get_landmark_points(landmarks)
    h, w = image.shape[:2]
    # 2. 建立五區遮罩
    masks = {}
    # 額頭
    masks['forehead'] = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 79:
        forehead_pts = np.array([points[i] for i in range(69, 79) if i < len(points)], dtype=np.int32)
        if len(forehead_pts) > 2:
            cv2.fillConvexPoly(masks['forehead'], forehead_pts, 255)
    else:
        if all(p in range(len(points)) for p in [19, 24, 27, 0, 16, 28]):
            face_width = points[16][0] - points[0][0]
            forehead_height_ref_y = (points[19][1] + points[24][1]) // 2
            forehead_height = abs(points[28][1] - forehead_height_ref_y)
            ellipse_center_y = forehead_height_ref_y - forehead_height
            cv2.ellipse(masks['forehead'], (points[27][0], ellipse_center_y), (int(face_width * 0.35), forehead_height), 0, 0, 360, 255, -1)
    # 左臉頰
    masks['left_cheek'] = np.zeros((h, w), dtype=np.uint8)
    if all(p in range(len(points)) for p in [1, 2, 3, 4, 31, 48]):
        left_cheek_pts = np.array([points[1], points[2], points[3], points[4], points[31], points[48]], dtype=np.int32)
        cv2.fillPoly(masks['left_cheek'], [left_cheek_pts], 255)
    # 右臉頰
    masks['right_cheek'] = np.zeros((h, w), dtype=np.uint8)
    if all(p in range(len(points)) for p in [15, 14, 13, 12, 35, 54]):
        right_cheek_pts = np.array([points[15], points[14], points[13], points[12], points[35], points[54]], dtype=np.int32)
        cv2.fillPoly(masks['right_cheek'], [right_cheek_pts], 255)
    # 鼻樑
    masks['nose'] = np.zeros((h, w), dtype=np.uint8)
    if all(p in range(len(points)) for p in [27, 28, 29, 30, 33]):
        nose_pts = np.array([points[27], points[28], points[29], points[30], points[33]], dtype=np.int32)
        cv2.fillConvexPoly(masks['nose'], nose_pts, 255)
    # 下巴
    masks['chin'] = np.zeros((h, w), dtype=np.uint8)
    if all(p in range(len(points)) for p in [6, 7, 8, 9, 57]):
        chin_pts = np.array([points[6], points[7], points[8], points[9], points[57]], dtype=np.int32)
        cv2.fillConvexPoly(masks['chin'], chin_pts, 255)
    # 3. 每區遮罩內膚色像素過濾 & LAB平均
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    means = {}
    for k in masks:
        filtered_mask = mask_skin_pixels(image, masks[k])
        if np.any(filtered_mask):
            mean_lab = cv2.mean(lab, mask=filtered_mask)[:3]
        else:
            mean_lab = (0,0,0)
        means[k] = mean_lab
    # 4. 加權平均（臉頰各20%、額頭20%、鼻樑20%、下巴20%）
    final_L = (means['left_cheek'][0] * 0.2 + means['right_cheek'][0] * 0.2 + means['forehead'][0] * 0.2 + means['nose'][0] * 0.2 + means['chin'][0] * 0.2)
    final_A = (means['left_cheek'][1] * 0.2 + means['right_cheek'][1] * 0.2 + means['forehead'][1] * 0.2 + means['nose'][1] * 0.2 + means['chin'][1] * 0.2)
    final_B = (means['left_cheek'][2] * 0.2 + means['right_cheek'][2] * 0.2 + means['forehead'][2] * 0.2 + means['nose'][2] * 0.2 + means['chin'][2] * 0.2)
    # 5. 直接用 OpenCV LAB（不做縮放/偏移）
    avg_lab = np.uint8([[[final_L, final_A, final_B]]])
    avg_bgr = cv2.cvtColor(avg_lab, cv2.COLOR_LAB2BGR)[0,0]
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
    # Debug log
    print("[膚色分析 debug]")
    for k in means:
        print(f"區域 {k} LAB: {means[k]}")
    print(f"加權平均 LAB: ({final_L}, {final_A}, {final_B})")
    print(f"最終 RGB: {avg_rgb}")
    return avg_rgb

def create_debug_image_dlib(image, face_rect, skin_mask, eye_masks, lip_mask, hair_mask, output_path):
    """使用 dlib 的遮罩建立高精度偵錯圖"""
    debug_img = image.copy()

    # 定義顏色 (BGR 格式)
    SKIN_COLOR = (0, 255, 255)  # Yellow
    EYE_COLOR = (255, 0, 0)      # Blue
    LIP_COLOR = (0, 0, 255)    # Red
    HAIR_COLOR = (42, 42, 165)   # Brown
    FACE_RECT_COLOR = (0, 255, 0) # Green

    # 疊加顏色遮罩
    # 皮膚
    if skin_mask is not None:
        skin_overlay = np.zeros_like(debug_img)
        skin_overlay[skin_mask == 255] = SKIN_COLOR
        cv2.addWeighted(debug_img, 1, skin_overlay, 0.4, 0, debug_img)
    # 眼睛
    if eye_masks:
        for eye_mask in eye_masks:
            eye_overlay = np.zeros_like(debug_img)
            eye_overlay[eye_mask == 255] = EYE_COLOR
            cv2.addWeighted(debug_img, 1, eye_overlay, 0.6, 0, debug_img)
    # 嘴唇
    if lip_mask is not None:
        lip_overlay = np.zeros_like(debug_img)
        lip_overlay[lip_mask == 255] = LIP_COLOR
        cv2.addWeighted(debug_img, 1, lip_overlay, 0.5, 0, debug_img)
    # 頭髮
    if hair_mask is not None:
        hair_overlay = np.zeros_like(debug_img)
        hair_overlay[hair_mask == 255] = HAIR_COLOR
        cv2.addWeighted(debug_img, 1, hair_overlay, 0.4, 0, debug_img)

    # 繪製臉部矩形框
    if face_rect:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), FACE_RECT_COLOR, 2)

    # 儲存偵錯圖
    cv2.imwrite(output_path, debug_img)
    print(f"✅ 偵錯圖已儲存至: {output_path}")

def create_debug_mask_image(image, skin_mask=None, eye_masks=None, lip_mask=None, hair_mask=None, output_path='debug_mask.png'):
    debug_img = image.copy()
    # 定義顏色 (BGR)
    SKIN_COLOR = (0, 255, 255)  # 黃
    EYE_COLOR = (255, 0, 0)     # 藍
    LIP_COLOR = (0, 0, 255)     # 紅
    HAIR_COLOR = (128, 0, 128)  # 紫
    # Debug: 輸出 hair_mask 的唯一值
    print('hair_mask unique:', np.unique(hair_mask) if hair_mask is not None else None)
    # 疊加遮罩
    if skin_mask is not None:
        skin_overlay = np.zeros_like(debug_img)
        skin_overlay[skin_mask == 255] = SKIN_COLOR
        cv2.addWeighted(debug_img, 1, skin_overlay, 0.4, 0, debug_img)
    if eye_masks:
        for eye_mask in eye_masks:
            eye_overlay = np.zeros_like(debug_img)
            eye_overlay[eye_mask == 255] = EYE_COLOR
            cv2.addWeighted(debug_img, 1, eye_overlay, 0.6, 0, debug_img)
    if lip_mask is not None:
        lip_overlay = np.zeros_like(debug_img)
        lip_overlay[lip_mask == 255] = LIP_COLOR
        cv2.addWeighted(debug_img, 1, lip_overlay, 0.5, 0, debug_img)
    if hair_mask is not None:
        hair_overlay = np.zeros_like(debug_img)
        hair_overlay[hair_mask == 255] = HAIR_COLOR
        cv2.addWeighted(debug_img, 1, hair_overlay, 0.4, 0, debug_img)
    cv2.imwrite(output_path, debug_img)
    print(f"✅ 遮罩圖已儲存至: {output_path}")

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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    prompt = (
        "請幫我判斷以下色碼的繁體中文名稱，回傳格式為 JSON 物件，key 為色碼，value 為中文名稱：\n"
        + ", ".join(hex_colors)
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    # 解析 Gemini 回傳的 JSON 內容
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    else:
        return {c: "未知色名" for c in hex_colors}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({"error": "請求中缺少圖片檔案"}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400

    # Save uploaded file
    filename = f"upload_{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    file.save(file_path)

    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "無法讀取圖片檔案"}), 500

    try:
        # --- 1. Face and Landmark Detection ---
        face, landmarks = detect_face_landmarks(image)
        if landmarks is None:
            return jsonify({"error": "未偵測到臉部特徵點，請使用更清晰的正面照。"}), 400

        # --- 2. Analyze Colors using new method ---
        skin_tone = calculate_skin_tone(image, landmarks)
        
        eye_masks = extract_eye_regions(image, landmarks)
        lip_mask = extract_lip_region(image, landmarks)
        hair_mask = extract_hair_mask_bisenet(image)
        if hair_mask is not None:
            debug_hair_mask_path = os.path.join(DEBUG_OUTPUT_DIR, f"hair_mask_{uuid.uuid4().hex}.png")
            cv2.imwrite(debug_hair_mask_path, hair_mask)
            print(f"hair_mask saved to: {debug_hair_mask_path}, unique: {np.unique(hair_mask)}")
        if hair_mask is None:
            hair_mask = extract_hair_region(image, landmarks, face)
            if hair_mask is not None:
                debug_hair_mask_path = os.path.join(DEBUG_OUTPUT_DIR, f"hair_mask_fallback_{uuid.uuid4().hex}.png")
                cv2.imwrite(debug_hair_mask_path, hair_mask)
                print(f"hair_mask (fallback) saved to: {debug_hair_mask_path}, unique: {np.unique(hair_mask)}")
        
        all_eye_colors = []
        if eye_masks:
            for eye_mask in eye_masks:
                all_eye_colors.extend(get_dominant_colors(image, mask=eye_mask))
        eye_color = all_eye_colors[0] if all_eye_colors else (128, 128, 128)

        lip_color = get_dominant_colors(image, mask=lip_mask)[0] if lip_mask is not None else (128, 128, 128)
        hair_color = get_dominant_colors(image, mask=hair_mask)[0] if hair_mask is not None else (128, 128, 128)

        # --- 3. Gemini 查詢色名 ---
        hex_colors = [
            rgb_to_hex(skin_tone),
            rgb_to_hex(eye_color),
            rgb_to_hex(hair_color),
            rgb_to_hex(lip_color)
        ]
        color_names = get_color_names_from_gemini(hex_colors)

        # --- 4. Seasonal Analysis ---
        season, analysis_details = analyze_seasonal_colors(skin_tone, eye_color, hair_color)

        # --- 5. Create Debug Image ---
        skin_mask_for_debug = extract_skin_region(image, landmarks)
        debug_image_filename = f"debug_{uuid.uuid4().hex}.jpg"
        debug_image_path = os.path.join(DEBUG_OUTPUT_DIR, debug_image_filename)
        create_debug_image_dlib(image, face, skin_mask_for_debug, eye_masks, lip_mask, hair_mask, debug_image_path)
        debug_image_url = f"/debug/{debug_image_filename}"

        # --- 5.1. Create Debug Mask Image (for front-end overlay) ---
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
            "debug_error": ""
        }
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] Critical error in dlib analyze route: {e}")
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
    app.run(host='0.0.0.0', port=5001)