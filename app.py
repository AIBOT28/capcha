import os
from flask import Flask, request, jsonify, render_template
from predict import predict

app = Flask(__name__)

# Cấu hình thư mục lưu ảnh upload tạm thời
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file ảnh.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tên file rỗng.'}), 400
        
    try:
        # Lưu file tạm thời
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Dự đoán
        prediction = predict(filepath)
        
        # Xóa file sau khi dự đoán xong (tùy chọn)
        try:
            os.remove(filepath)
        except Exception:
            pass
            
        if prediction is None:
             return jsonify({'success': False, 'error': 'Không thể dự đoán. Lỗi tải mô hình.'}), 500
             
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0')
