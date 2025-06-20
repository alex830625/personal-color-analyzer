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

請提供具體的穿搭建議、彩妝建議，以及為什麼這些顏色適合這個季節型的原因。語氣要親切、專業且實用。敘述的內容中不要顯示色碼，直接以該色碼的繁體中文顯示，並且分段落呈現，例如肌膚色調分析、頭髮顏色分析...等，分析說明如以下格式 :您的頭髮顏色為（如：深棕、淺褐、黑色等），與膚色搭配（如：協調／對比），整體感覺（如：自然、成熟、活潑等，開頭部分不需要親愛的，也不需要甚麼某某型美人，清楚且直接的說明即可`;

    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: prompt,
    });
    const suggestion = response.text;
    res.json({ suggestion });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Gemini 產生建議失敗', details: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});