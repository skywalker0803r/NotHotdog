from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model import HotdogClassifier
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化模型
model = HotdogClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '沒有檔案'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 進行預測
        is_hotdog, confidence = model.predict(filepath)
        
        # 刪除暫存檔案
        os.remove(filepath)
        
        return jsonify({
            'is_hotdog': bool(is_hotdog),
            'confidence': float(confidence)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)