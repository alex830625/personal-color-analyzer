import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import axios from 'axios';
import FormData from 'form-data';
import dotenv from 'dotenv';
import { GoogleGenAI } from "@google/genai";
dotenv.config();


const app = express();
const PORT = 5000;
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/api/upload', upload.single('photo'), async (req, res) => {
  try {
    const filePath = req.file.path;
    const pythonApiUrl = 'http://localhost:5001/analyze';

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const response = await axios.post(pythonApiUrl, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    fs.unlinkSync(filePath);
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: '分析失敗', details: err.message });
  }
});


app.post('/api/gemini-suggestion', async (req, res) => {
  try {
    const { season, season_name, skin_tone, eye_color, hair_color, color_suggestions } = req.body;
    
    // 構建更詳細的提示詞
    const prompt = `你是一位專業的個人色彩顧問。根據以下詳細分析結果，請用繁體中文寫一段人化色彩建議，內容包含：

季節型：${season_name} (${season})
膚色：${skin_tone}
眼睛顏色：${eye_color}
頭髮顏色：${hair_color}

建議色板：
- 服裝：${color_suggestions.clothes.join('、')}
- 彩妝：${color_suggestions.makeup.join('、')}
- 珠寶：${color_suggestions.jewelry.join('、')}
- 避免顏色：${color_suggestions.avoid.join('、')}

請用 markdown 格式撰寫，所有小標題請用 ## 或 ### 標記。
請提供具體的穿搭建議、彩妝建議，以及為什麼這些顏色適合這個季節型的原因。語氣要親切、專業且實用，不要提到性別。敘述的內容中不要顯示色碼，直接以下方提供的色名描述，並且分段落呈現，例如肌膚色調分析、頭髮顏色分析...等。**請務必只用上述膚色、眼睛、頭髮的色名（${skin_tone}、${eye_color}、${hair_color}），不要自行創造或改寫顏色名稱。** 分析說明如以下格式 :1. 👤 色彩季型判定（春暖／夏冷／秋暖／冬冷，並解釋理由）
2. 👗 穿搭建議色彩（分為主色、配色、避免色）
3. 💄 化妝品建議（底妝質地、腮紅色系、唇色、眼影）
4. 💍 飾品建議（金屬色系與風格）
段落前面不要有數字`;

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: prompt,
    });
    const suggestion = response.text;
    res.json({ suggestion });
  } catch (err) {
    res.status(500).json({ error: 'Gemini 產生建議失敗', details: err.message });
  }
});

app.post('/api/color-names', async (req, res) => {
  try {
    const { hexes } = req.body;
    if (!Array.isArray(hexes) || hexes.length === 0) {
      return res.status(400).json({ error: '缺少 hexes 陣列' });
    }
    // 組 prompt
    const prompt = `請將以下 HEX 色碼轉為繁體中文顏色名稱，回傳 JSON 格式（key 為色碼，value 為繁體中文名稱）：\n${JSON.stringify(hexes)}\n只回傳 JSON，不要多餘說明。`;
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: prompt,
    });
    // 嘗試解析 Gemini 回傳的 JSON
    let colorNames = {};
    try {
      // 只取出第一個 { ... } JSON 區塊
      const match = response.text.match(/\{[\s\S]*\}/);
      if (match) {
        colorNames = JSON.parse(match[0]);
      }
    } catch (e) {
      return res.status(500).json({ error: 'Gemini 回傳格式解析失敗', details: e.message, raw: response.text });
    }
    res.json(colorNames);
  } catch (err) {
    res.status(500).json({ error: 'Gemini 取得色名失敗', details: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});