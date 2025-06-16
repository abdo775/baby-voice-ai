import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from pydub import AudioSegment
from tensorflow.keras.models import load_model  # type: ignore
import random

app = Flask(__name__)

# ========== Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "baby_cry_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Load Resources ==========
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing files: {e}")

# ========== Prediction Messages ==========
prediction_map = {
    "3": [...],  # نفس القوائم اللي عندك اختصرناها هنا للاختصار
    "1": [...],
    "2": [...],
    "0": [...],
    "4": [...]
}

# ========== Routes ==========
@app.route('/')
def index():
    return '🎧 Baby Cry Prediction API is running! Use POST /predict with an audio file.'

# ========== Prediction Helper ==========
def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    features = np.array(list(mfccs_mean) + list(mfccs_std) + [
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        zero_crossing_rate, rms, chroma_stft]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

    prediction = model.predict(features_scaled)
    predicted_index = str(np.argmax(prediction))
    response = random.choice(prediction_map[predicted_index])

    return response, predicted_index

# ========== Prediction Endpoint ==========
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file provided."}), 400

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.wav', '.webm', '.mp4', '.mp3', '.m4a', '.ogg', '.aac', '.flac']:
            return jsonify({"error": f"Unsupported file format: {file_ext}"}), 400

        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_path)

        if file_ext != '.wav':
            try:
                sound = AudioSegment.from_file(temp_path)
                wav_path = temp_path.rsplit('.', 1)[0] + ".wav"
                sound.export(wav_path, format="wav")
                os.remove(temp_path)
                temp_path = wav_path
            except Exception as e:
                return jsonify({"error": f"Audio conversion error: {str(e)}"}), 500

        prediction_text, prediction_label = predict_audio(temp_path)
        os.remove(temp_path)  # Clean up uploaded file after prediction

        return jsonify({
            "prediction": prediction_text,
            "class": prediction_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Run App ==========
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
