import os
import base64
import warnings
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import json

# Nonaktifkan warning
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path model dan label
MODEL_PATH = os.path.join('models', 'skin_analysis_cnn.h5')
LABEL_PATH = os.path.join('models', 'labels.json')

# Load model dan label
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

try:
    with open(LABEL_PATH, 'r') as f:
        label_map = json.load(f)
        CONDITION_LABELS = sorted(label_map, key=lambda k: label_map[k])
except:
    CONDITION_LABELS = ['acne', 'dry', 'normal', 'oily']
    print("⚠️ Using default labels:", CONDITION_LABELS)

# Skincare Recommendation
SKINCARE_RECOMMENDATIONS = {
    "acne": {"cleanser": "Acne Control Cleanser", "moisturizer": "Non-comedogenic Moisturizer"},
    "dry": {"cleanser": "Hydrating Cleanser", "moisturizer": "Deep Moisture Cream"},
    "oily": {"cleanser": "Oil-Free Cleanser", "moisturizer": "Mattifying Gel Moisturizer"},
    "normal": {"cleanser": "Balanced Cleanser", "moisturizer": "Daily Moisturizer"}
}

def preprocess_image(img, target_size=(224, 224)):
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def analyze_with_cnn(img):
    if model is None:
        return None
    processed = preprocess_image(img)
    preds = model.predict(processed)[0]
    class_idx = int(np.argmax(preds))
    return {
        "condition": CONDITION_LABELS[class_idx],
        "confidence": float(np.max(preds)),
        "all_predictions": {label: float(prob) for label, prob in zip(CONDITION_LABELS, preds)}
    }

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('.', path)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Ambil file atau base64 image
        if 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
        elif request.is_json and 'image' in request.json:
            img_base64 = request.json['image'].split(',')[1]
            img_bytes = base64.b64decode(img_base64)
        else:
            return jsonify({'error': 'No image provided'}), 400

        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Analisis
        result = analyze_with_cnn(img)
        if result:
            return jsonify({
                'image': base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'),
                'analysis_method': 'CNN',
                'condition': result['condition'],
                'confidence': round(result['confidence'] * 100, 2),
                'recommendations': SKINCARE_RECOMMENDATIONS.get(result['condition'], {}),
                'all_predictions': result['all_predictions'],
                'after_image': None
            })
        else:
            return jsonify({'error': 'Model not available'}), 500

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
