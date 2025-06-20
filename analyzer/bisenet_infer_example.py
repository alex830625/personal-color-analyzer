import cv2
import numpy as np
import torch
import face_recognition
from bisenet import load_bisenet_model

# 參數
IMAGE_PATH = 'input.jpg'  # 測試圖片
WEIGHT_PATH = '79999_iter.pth'  # BiSeNet 預訓練權重
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 載入圖片與偵測人臉
image = cv2.imread(IMAGE_PATH)
face_locations = face_recognition.face_locations(image)
if not face_locations:
    raise Exception('找不到人臉')
top, right, bottom, left = face_locations[0]
face_img = image[top:bottom, left:right]

# 2. 載入 BiSeNet 模型
model = load_bisenet_model(WEIGHT_PATH, device=DEVICE, n_classes=19)

# 3. 預處理並推論
face_resized = cv2.resize(face_img, (512, 512))
face_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) / 255.0
with torch.no_grad():
    out = model(face_tensor)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)  # shape: (512, 512)

# 4. 取出頭髮區域（class 17）
hair_mask = (parsing == 17).astype(np.uint8) * 255

# 5. 還原到原圖座標
hair_mask_resized = cv2.resize(hair_mask, (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
hair_mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
hair_mask_full[top:bottom, left:right] = hair_mask_resized

# 6. 取出頭髮像素並分析顏色
hair_only = cv2.bitwise_and(image, image, mask=hair_mask_full)
hair_pixels = image[hair_mask_full == 255]
if len(hair_pixels) == 0:
    print('未偵測到頭髮像素')
else:
    mean_color = np.mean(hair_pixels, axis=0)
    print('頭髮平均色(BGR):', mean_color)
    # 你可以進一步轉成 HSV 或其他色彩空間

# 7. 可選：儲存頭髮遮罩與結果
cv2.imwrite('hair_mask.png', hair_mask_full)
cv2.imwrite('hair_only.png', hair_only) 