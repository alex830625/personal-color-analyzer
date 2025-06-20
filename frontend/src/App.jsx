import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

// 常見色碼對應中文名稱（擴充）
const colorNameMap = {
  '#FFFFFF': '白色',
  '#000000': '黑色',
  '#FFD700': '金黃色',
  '#FFA07A': '淺鮭紅',
  '#FFB347': '橙黃色',
  '#98FB98': '淡綠色',
  '#FF69B4': '亮粉紅',
  '#FFB6C1': '淺粉紅',
  '#FFDAB9': '桃色',
  '#8B4513': '深褐色',
  '#A0522D': '赭色',
  '#DEB887': '淺棕色',
  '#B0E0E6': '粉藍色',
  '#E0FFFF': '淡青色',
  '#C1C1FF': '淡紫色',
  '#DB7093': '玫瑰紅',
  '#B0C4DE': '亮鋼藍',
  '#4682B4': '鋼藍色',
  '#708090': '灰藍色',
  '#B8860B': '深金色',
  '#4169E1': '寶藍色',
  '#DC143C': '深紅色',
  '#00CED1': '深青色',
  '#C0C0C0': '銀色',
  '#ADD8E6': '亮藍色',
  '#BDB76B': '橄欖黃',
  '#D2691E': '巧克力色',
  '#D2B48C': '卡其色',
  '#E6E6FA': '薰衣草紫',
  '#8A2BE2': '藍紫色',
  '#FF8C00': '深橙色',
  '#00BFFF': '亮天藍',
  '#E0FFFF': '亮青色',
  '#F5DEB3': '小麥色',
  '#F08080': '淺珊瑚紅',
  '#FA8072': '鮭紅',
  '#F4A460': '沙棕色',
  '#D8BFD8': '蓟色',
  '#DDA0DD': '梅紫色',
  '#EEE8AA': '淺卡其',
  '#F0E68C': '卡其黃',
  '#E9967A': '深鮭紅',
  '#8FBC8F': '暗海綠',
  '#20B2AA': '亮海藍',
  '#5F9EA0': '軍藍色',
  '#BC8F8F': '玫瑰棕',
  '#CD5C5C': '印度紅',
  '#A9A9A9': '深灰色',
  '#808080': '灰色',
  '#D3D3D3': '亮灰色',
  '#F8F8FF': '幽靈白',
  '#FFE4E1': '薄霧玫瑰',
  '#FFF0F5': '薰衣草紅',
  '#F0FFF0': '蜜瓜綠',
  '#F5FFFA': '薄荷奶油',
  '#F0FFFF': '天藍',
  '#F0F8FF': '愛麗絲藍',
  '#FFF5EE': '海貝色',
  '#FFE4B5': '摩卡色',
  '#FFEBCD': '奶油色',
  '#FFEFD5': '番木色',
  '#FFF8DC': '玉米色',
  '#FFFACD': '檸檬色',
  '#FAFAD2': '淺卡其',
  '#FFFFE0': '亮黃',
  '#FFFFF0': '象牙白',
  '#FDF5E6': '老蕾絲',
  '#FFFAFA': '雪白',
  '#F5F5F5': '白煙',
  '#F5F5DC': '米色',
  '#F5DEB3': '小麥色',
  '#FFF8DC': '玉米色',
  '#FFE4C4': '淺棕',
  '#FFDAB9': '桃色',
  '#EEE8AA': '淺卡其',
  '#B22222': '磚紅',
  '#FF6347': '番茄紅',
  '#FF4500': '橘紅',
  '#DA70D6': '蘭花紫',
  '#BA55D3': '紫羅蘭',
  '#9932CC': '暗紫羅蘭',
  '#9400D3': '紫色',
  '#8B008B': '深洋紅',
  '#800080': '紫紅',
  '#4B0082': '靛藍',
  '#6A5ACD': '板岩藍',
  '#483D8B': '暗板岩藍',
  '#191970': '午夜藍',
  '#00008B': '深藍',
  '#000080': '海軍藍',
  '#008080': '水鴨色',
  '#008B8B': '暗青色',
  '#2F4F4F': '深石板灰',
  '#556B2F': '暗橄欖綠',
  '#6B8E23': '橄欖綠',
  '#808000': '橄欖色',
  '#BDB76B': '橄欖黃',
  '#F0E68C': '卡其黃',
  '#FFD700': '金黃色',
  '#FFA500': '橙色',
  '#FF8C00': '深橙色',
  '#FF7F50': '珊瑚色',
  '#FF6347': '番茄紅',
  '#FF4500': '橘紅',
  '#FFDAB9': '桃色',
  '#EEE8AA': '淺卡其',
  '#F5DEB3': '小麥色',
  '#FFF8DC': '玉米色',
  '#FFE4C4': '淺棕',
};

// HEX 轉 RGB
function hexToRgb(hex) {
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const num = parseInt(c, 16);
  return [num >> 16, (num >> 8) & 0xff, num & 0xff];
}
// RGB 轉 HEX
function rgbToHex([r, g, b]) {
  return (
    '#' +
    r.toString(16).padStart(2, '0') +
    g.toString(16).padStart(2, '0') +
    b.toString(16).padStart(2, '0')
  ).toUpperCase();
}
// 計算兩個 HEX 顏色的歐式距離
function colorDistance(hex1, hex2) {
  const rgb1 = hexToRgb(hex1);
  const rgb2 = hexToRgb(hex2);
  return Math.sqrt(
    Math.pow(rgb1[0] - rgb2[0], 2) +
    Math.pow(rgb1[1] - rgb2[1], 2) +
    Math.pow(rgb1[2] - rgb2[2], 2)
  );
}
// 取得最相近的顏色名稱
function getColorName(hex) {
  const code = hex?.toUpperCase();
  if (colorNameMap[code]) return colorNameMap[code];
  // 找最相近的顏色
  let minDist = Infinity;
  let closest = '自訂色';
  for (const [k, v] of Object.entries(colorNameMap)) {
    const dist = colorDistance(code, k);
    if (dist < minDist) {
      minDist = dist;
      closest = v;
    }
  }
  return closest;
}

const seasonNameMap = {
  spring: '春季',
  summer: '夏季',
  autumn: '秋季',
  winter: '冬季',
};

function ColorBox({ color }) {
  return (
    <div className="w-8 h-8 rounded shadow border m-1" style={{ backgroundColor: color }} title={color}></div>
  );
}

function cleanSuggestion(text) {
  if (!text) return '';
  // 去除開頭和結尾的單/雙引號（只要有一邊就去掉）
  text = text.replace(/^['"]+|['"]+$/g, '');
  // 將 \\n、\r\n、\n、\r 都換成 \n
  text = text.replace(/\\r\\n|\\n|\\r/g, '\n').replace(/\r\n|\r/g, '\n');
  // 處理多餘的 \\\\n
  text = text.replace(/\\\\n/g, '\n');
  // 將所有 \n 轉成真正的換行
  text = text.replace(/\\n/g, '\n');
  // 切成段落（兩個以上換行分段）
  let paragraphs = text.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  // 去除重複段落
  let seen = new Set();
  let result = [];
  for (let p of paragraphs) {
    if (!seen.has(p)) {
      seen.add(p);
      result.push(p);
    }
  }
  return result.join('\n\n');
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestion, setSuggestion] = useState('');
  const [rawResponse, setRawResponse] = useState(null);

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError('');
      setSuggestion('');
      setRawResponse(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);
    setSuggestion('');
    setRawResponse(null);
    
    const formData = new FormData();
    formData.append('photo', file);

    try {
      const res = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      setRawResponse(data);
      
      if (res.ok) {
        setResult(data);
        // 呼叫 GPT API
        setSuggestion('');
        try {
          const gptRes = await fetch('http://localhost:5000/api/gemini-suggestion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              season: data.season,
              season_name: data.season_name,
              skin_tone: data.color_names?.skin_tone || colorNameByHex(data.skin_tone, 'skin_tone'),
              eye_color: data.color_names?.eye_color || colorNameByHex(data.eye_color, 'eye_color'),
              hair_color: data.color_names?.hair_color || colorNameByHex(data.hair_color, 'hair_color'),
              color_suggestions: data.color_suggestions,
            }),
          });
          if (gptRes.ok) {
            const gptData = await gptRes.json();
            setSuggestion(gptData.suggestion);
          } else {
            setSuggestion('無法取得建議，請稍後再試');
          }
        } catch {
          console.log('GEMINI_API_KEY:', process.env.GEMINI_API_KEY);
          setSuggestion('無法取得建議，請稍後再試');
        }
      } else {
        setError(data.error || '分析失敗，請檢查後端服務');
      }
    } catch (err) {
      setError('連線失敗，請檢查分析器服務是否運行中。');
      setRawResponse({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  // 修改 colorNameByHex，優先用 result.color_names
  function colorNameByHex(hex, type) {
    if (!hex) return '';
    if (result && result.color_names && type && result.color_names[type]) return result.color_names[type];
    const code = hex.toLowerCase();
    if (result && result.color_names && result.color_names[code]) return result.color_names[code];
    if (result && result.color_names && result.color_names[hex]) return result.color_names[hex];
    return getColorName(hex);
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-indigo-100 via-pink-100 to-blue-100 px-2">
      <div className="w-full max-w-xl flex flex-col items-center">
        <h1 className="text-4xl font-extrabold tracking-wide mb-8 text-black text-center drop-shadow-lg">個人色彩分析</h1>
        <div className="w-full bg-white rounded-2xl shadow-2xl p-8 flex flex-col items-center">
          <label htmlFor="photo-upload" className="mb-6 cursor-pointer inline-block px-6 py-3 bg-gradient-to-r from-indigo-400 to-pink-400 text-white font-semibold rounded-lg shadow-md hover:from-pink-400 hover:to-indigo-400 transition-all duration-300 text-lg">
            選擇檔案
            <input id="photo-upload" type="file" accept="image/*" onChange={handlePhotoChange} className="hidden" />
          </label>
          {preview && <img src={preview} alt="預覽" className="w-40 h-40 object-cover rounded-xl mb-6 shadow-md border-2 border-indigo-200" />}
          <button onClick={handleAnalyze} disabled={!file || loading} className="w-full py-3 rounded-lg text-lg font-bold bg-gradient-to-r from-blue-400 to-indigo-400 text-white shadow-lg hover:from-indigo-400 hover:to-blue-400 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed">
            {loading ? (
              <>
                <span className="spinner"></span>分析中...
              </>
            ) : (
              '上傳並分析'
            )}
          </button>
        </div>

        {/* 結果與失敗反饋區塊 */}
        {!loading && (result || error) && (
          <div className="mt-8 p-6 bg-white rounded-lg shadow-xl animate-fade-in text-gray-800 w-full">
            {result ? (
              // 成功顯示結果
              <>
                <h2 className="text-xl font-semibold mb-2 text-center">分析結果</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 mb-4">
                  <div className="flex items-center">
                    肌膚色調：
                    <div className="w-8 h-8 rounded shadow border mx-2 hover:scale-110 hover:ring-2 hover:ring-blue-300 transition-all duration-200" style={{ backgroundColor: result.skin_tone }} />
                    <span className="font-bold">{colorNameByHex(result.skin_tone, 'skin_tone')}</span>
                  </div>
                  <div className="flex items-center">
                    眼睛顏色：
                    <div className="w-8 h-8 rounded shadow border mx-2 hover:scale-110 hover:ring-2 hover:ring-blue-300 transition-all duration-200" style={{ backgroundColor: result.eye_color }} />
                    <span className="font-bold">{colorNameByHex(result.eye_color, 'eye_color')}</span>
                  </div>
                  <div className="flex items-center">
                    頭髮顏色：
                    <div className="w-8 h-8 rounded shadow border mx-2 hover:scale-110 hover:ring-2 hover:ring-blue-300 transition-all duration-200" style={{ backgroundColor: result.hair_color }} />
                    <span className="font-bold">{colorNameByHex(result.hair_color, 'hair_color')}</span>
                  </div>
                  <div className="flex items-center">
                    季節性色彩：
                    <div className="w-8 h-8 rounded shadow border mx-2 hover:scale-110 hover:ring-2 hover:ring-blue-300 transition-all duration-200" style={{ backgroundColor: result.color_suggestions?.clothes?.[0] || '#fff' }} />
                    <span className="font-bold">{result.season_name || seasonNameMap[result.season] || result.season}</span>
                  </div>
                </div>
                <div className="mb-2">衣服顏色建議：
                  <div className="flex flex-wrap mt-1">
                    {result.color_suggestions?.clothes?.map((c, i) => <div key={i} className="flex flex-col items-center mr-2 mb-2"><ColorBox color={c} /><span className="text-xs mt-1">{colorNameByHex(c)}</span></div>)}
                  </div>
                </div>
                <div className="mb-2">化妝顏色建議：
                  <div className="flex flex-wrap mt-1">
                    {result.color_suggestions?.makeup?.map((c, i) => <div key={i} className="flex flex-col items-center mr-2 mb-2"><ColorBox color={c} /><span className="text-xs mt-1">{colorNameByHex(c)}</span></div>)}
                  </div>
                </div>
                <div className="mb-2">珠寶&飾品建議：
                  <div className="flex flex-wrap mt-1">
                    {result.color_suggestions?.jewelry?.map((c, i) => <div key={i} className="flex flex-col items-center mr-2 mb-2"><ColorBox color={c} /><span className="text-xs mt-1">{colorNameByHex(c)}</span></div>)}
                  </div>
                </div>
                <div className="mb-2">避免使用的顏色：
                  <div className="flex flex-wrap mt-1">
                    {result.color_suggestions?.avoid?.map((c, i) => <div key={i} className="flex flex-col items-center mr-2 mb-2"><ColorBox color={c} /><span className="text-xs mt-1">{colorNameByHex(c)}</span></div>)}
                  </div>
                </div>
                <div className="mt-4 p-4 bg-blue-50 rounded text-black min-h-[60px]">
                  <div className="font-bold mb-2">AI 個人化色彩建議：</div>
                  <ReactMarkdown
                    components={{
                      h2: ({node, ...props}) => <h2 style={{fontSize: '1.5rem', fontWeight: 'bold', marginTop: '1em', marginBottom: '1em'}} {...props} />,
                      h3: ({node, ...props}) => <h3 style={{fontSize: '1.2rem', fontWeight: 'bold', marginTop: '1em', marginBottom: '0.5em'}} {...props} />,
                      p: ({node, ...props}) => <p style={{marginTop: '1em', marginBottom: '1em'}} {...props} />
                    }}
                  >
                    {cleanSuggestion(suggestion)}
                  </ReactMarkdown>
                </div>
              </>
            ) : (
              // 失敗顯示反饋
              <div>
                <h3 className="text-xl font-semibold text-red-600 mb-4">分析失敗</h3>
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  {preview && <img src={preview} alt="上傳的圖片" className="w-40 h-40 object-cover rounded shadow-md"/>}
                  <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 w-full" role="alert">
                    <p className="font-bold">錯誤訊息：</p>
                    <p>{error}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
