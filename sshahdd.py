import numpy as np
import librosa
import joblib
import tensorflow as tf
import argparse  # استيراد argparse

# إعداد argparse لاستقبال المسار من الـ Terminal
parser = argparse.ArgumentParser(description="Test a baby cry sound using the trained model.")
parser.add_argument('file_path', type=str, help="Path to the audio file to test")
args = parser.parse_args()

# تحميل النموذج والمحولين
model = tf.keras.models.load_model(r'C:\Users\it\Downloads\Project\baby_cry_model.h5')
encoder = joblib.load(r'C:\Users\it\Downloads\Project\label_encoder.pkl')
scaler = joblib.load(r'C:\Users\it\Downloads\Project\scaler.pkl')

# استخدام المسار المدخل من سطر الأوامر
file_path = args.file_path

y, sr = librosa.load(file_path, sr=None)

# استخراج الخصائص
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
rms = librosa.feature.rms(y=y).mean()
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_std = np.std(mfccs, axis=1)

# جمع الخصائص في مصفوفة واحدة
features = np.array(list(mfccs_mean) + list(mfccs_std) + [
    spectral_centroid, spectral_bandwidth, spectral_rolloff,
    zero_crossing_rate, rms, chroma_stft]).reshape(1, -1)

# تطبيع وتهيئة البيانات
features_scaled = scaler.transform(features)
features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

# التنبؤ
prediction = model.predict(features_scaled)
predicted_label_index = np.argmax(prediction)

# تحويل الرقم إلى اسم الفئة
predicted_label = encoder.inverse_transform([predicted_label_index])

# طباعة اسم الفئة المتنبأ بها
print(f"Predicted class: {predicted_label[0]}")
