import os
import re
import random
import pandas as pd
from pydub import AudioSegment
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import whisper
import nltk
import joblib

nltk.download('punkt')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# === Настройки ===
INPUT_DIR = os.path.join(BASE_DIR, "conversations")
OUTPUT_DIR = os.path.join(BASE_DIR, "app/models")
AUGMENTED_DIR = os.path.join(BASE_DIR, "conversations_augmented")

print(f"Корневая папка проекта: {BASE_DIR}")
print(f"Путь к данным: {INPUT_DIR}")
print(f"Путь к результатам: {OUTPUT_DIR}")
print(f"Путь к аугментированным данным: {AUGMENTED_DIR}")

keywords = {"покупка", "заказ", "доставка", "оплата", "услуга"}
neutral_phrases = ["угу", "ясно", "понятно", "спасибо", "ага", "хорошо", "ладно", "так"]

# === Whisper Model (base) ===
print("🔄 Загрузка модели Whisper (base)...")
model = whisper.load_model("base")
print("✅ Модель Whisper (base) загружена!")

def recognize(mp3_path):
    try:
        result = model.transcribe(mp3_path, language="ru")
        text = result["text"]
        print(f"🗣 Распознанный текст: {text}")
        return text
    except Exception as e:
        print(f"❌ Ошибка распознавания {mp3_path}: {e}")
        return ""

def extract_features(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    duration_est = len(words)
    kw_count = sum(1 for w in words if w in keywords)
    return {
        "duration_sec": duration_est,
        "keywords_count": kw_count,
    }

# === Обработка данных ===
records = []

label_map = {
    "hot": "Горячий",
    "warm": "Тёплый",
    "cold": "Холодный"
}

for label_folder, label_name in label_map.items():
    folder_path = os.path.join(INPUT_DIR, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            path = os.path.join(folder_path, filename)
            print(f"🔍 Обрабатываем файл: {filename}")

            original_text = recognize(path)
            if original_text:
                feats = extract_features(original_text)
                feats["interest_level"] = label_name
                records.append(feats)

print(f"Количество записей: {len(records)}")
if len(records) == 0:
    print("⚠️ Нет данных для обучения!")
    exit(1)

# === Обучение модели ===
df = pd.DataFrame(records)
X = df.drop("interest_level", axis=1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["interest_level"])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False)
model.fit(X_train, y_train)

# === Оценка ===
y_pred = model.predict(X_test)
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Сохранение ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgboost_call_model.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
print("\n✅ Обучение завершено и модель сохранена.")

