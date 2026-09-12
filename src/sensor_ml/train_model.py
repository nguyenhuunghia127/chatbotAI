# -*- coding: utf-8 -*-
"""
MODULE HUẤN LUYỆN MÔ HÌNH RANDOM FOREST CHO EDGE AI (LOCAL ML)
Dự án: Hệ thống Giám sát Người cao tuổi Nội bộ
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Mục tiêu:
- Huấn luyện mô hình Scikit-learn (RandomForestClassifier) nhẹ, tiêu tốn ít RAM/CPU
- Phân loại đa lớp sự kiện:
  + Normal (Bình thường)
  + Fall (Té ngã khẩn cấp)
  + Fever (Sốt cao / Viêm nhiễm)
  + Cardiac_Alert (Cảnh báo nhịp tim bất thường)
- Lưu mô hình và metadata vào thư mục models/ dưới dạng .joblib
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Đảm bảo tương thích UTF-8 cho console trên Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Các đặc trưng đầu vào (Input Features) từ cảm biến
FEATURE_COLUMNS = [
    "aspect_ratio",        # Tỉ lệ H/W khung xương từ Camera (MediaPipe)
    "torso_angle",         # Góc nghiêng thân người (0-90 độ)
    "vertical_velocity",   # Vận tốc rơi trục Y (m/s)
    "sound_db",            # Độ ồn âm thanh phòng (dB)
    "body_temp",           # Thân nhiệt (°C)
    "heart_rate"           # Nhịp tim từ vòng BLE (BPM)
]

TARGET_COLUMN = "ground_truth_label"

# Đường dẫn mặc định
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sensor_rf_model.joblib")
DATA_FILE = os.path.join(PROJECT_ROOT, "mock_sensor_data.txt")


def generate_augmented_training_data(base_df: pd.DataFrame, num_samples: int = 400) -> pd.DataFrame:
    """
    Kỹ thuật Tăng cường dữ liệu (Data Augmentation) trên Edge:
    Từ dữ liệu mock cơ sở, sinh thêm các mẫu có nhiễu Gaussian nhẹ để mô hình
    Random Forest học tốt hơn các biên quyết định (decision boundaries)
    mà không bị overfitting hoặc thiếu dữ liệu.
    """
    augmented_rows = []
    labels = base_df[TARGET_COLUMN].unique()
    
    samples_per_label = num_samples // len(labels)
    
    for label in labels:
        subset = base_df[base_df[TARGET_COLUMN] == label]
        for _ in range(samples_per_label):
            # Chọn ngẫu nhiên 1 hàng mẫu làm hạt giống
            base_row = subset.sample(n=1).iloc[0]
            
            # Thêm nhiễu ngẫu nhiên nhỏ (Sensor Noise)
            noise_ratio = np.random.normal(0, 0.04)
            noise_angle = np.random.normal(0, 1.5)
            noise_vel = np.random.normal(0, 0.08)
            noise_sound = np.random.normal(0, 1.2)
            noise_temp = np.random.normal(0, 0.1)
            noise_hr = np.random.normal(0, 2.0)
            
            row = {
                "aspect_ratio": max(0.1, round(float(base_row["aspect_ratio"]) + noise_ratio, 2)),
                "torso_angle": max(0.0, min(90.0, round(float(base_row["torso_angle"]) + noise_angle, 1))),
                "vertical_velocity": round(float(base_row["vertical_velocity"]) + noise_vel, 2),
                "sound_db": max(20.0, round(float(base_row["sound_db"]) + noise_sound, 1)),
                "body_temp": round(float(base_row["body_temp"]) + noise_temp, 1),
                "heart_rate": int(max(40, round(float(base_row["heart_rate"]) + noise_hr))),
                TARGET_COLUMN: label
            }
            augmented_rows.append(row)
            
    return pd.DataFrame(augmented_rows)


def train_sensor_model(data_path: str = DATA_FILE, save_path: str = MODEL_PATH) -> RandomForestClassifier:
    """
    Hàm chính huấn luyện mô hình Scikit-Learn:
    - Tiết kiệm tài nguyên: n_estimators=50, max_depth=8 (rất nhẹ cho Orange Pi 5 / PC)
    - Tốc độ suy luận: < 1ms / mẫu
    """
    print("=" * 70, flush=True)
    print("[EDGE AI TRAINING] KHOI DONG TIEN TRINH HUAN LUYEN MO HINH CAM BIEN", flush=True)
    print("=" * 70, flush=True)

    try:
        # 1. Đọc dữ liệu mẫu
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Khong tim thay file du lieu tai: {data_path}")

        print(f"[DATA] Dang doc du lieu cam bien tu: {data_path}", flush=True)
        # Đọc bỏ qua các dòng comment bắt đầu bằng '#'
        df_base = pd.read_csv(data_path, comment="#")
        print(f"[OK] Da tai {len(df_base)} dong du lieu co so.", flush=True)

        # 2. Sinh dữ liệu tăng cường (Augmentation)
        df_train = generate_augmented_training_data(df_base, num_samples=600)
        print(f"[AUGMENTATION] Tap du lieu sau khi tang cuong: {len(df_train)} mau.", flush=True)

        X = df_train[FEATURE_COLUMNS]
        y = df_train[TARGET_COLUMN]

        # 3. Phân chia tập huấn luyện / kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 4. Khởi tạo mô hình Random Forest tối ưu cho Embedded / Edge
        # n_estimators=50: đủ cây quyết định nhưng không tốn RAM
        # max_depth=8: chống overfitting và đảm bảo tốc độ dự đoán thời gian thực
        # n_jobs=1: an toàn tuyệt đối trên Windows & Edge/SBC (tránh xung đột IPC process pool)
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            n_jobs=1
        )

        print("[TRAINING] Dang huan luyen Random Forest...", flush=True)
        model.fit(X_train, y_train)

        # 5. Đánh giá độ chính xác
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[EVALUATION] Do chinh xac tren tap kiem thu: {acc * 100:.2f}%", flush=True)
        print("\n--- BAO CAO PHAN LOAI CHI TIET ---", flush=True)
        print(classification_report(y_test, y_pred), flush=True)

        # 6. Đóng gói và lưu trữ mô hình
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_payload = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": list(model.classes_),
            "version": "1.0.0"
        }
        joblib.dump(model_payload, save_path)
        print(f"[SAVE] Da luu mo hinh thanh cong tai: {save_path}", flush=True)
        print("=" * 70, flush=True)
        return model

    except Exception as e:
        print(f"[ERROR HUAN LUYEN]: {str(e)}", file=sys.stderr, flush=True)
        raise e


if __name__ == "__main__":
    train_sensor_model()
