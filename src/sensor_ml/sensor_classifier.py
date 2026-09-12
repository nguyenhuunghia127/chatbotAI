# -*- coding: utf-8 -*-
"""
MODULE PHÂN LOẠI CẢM BIẾN & TỰ ĐỘNG GHI LOG NHẬT KÝ HỆ THỐNG
Dự án: Hệ thống Giám sát Người cao tuổi Nội bộ (Offline 100%)
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Chức năng:
1. Nạp mô hình Scikit-learn (Random Forest) từ file joblib (tự động huấn luyện nếu chưa có).
2. Phân loại đa luồng cảm biến (Sensor Fusion):
   - Camera: Tọa độ khung xương, tỉ lệ H/W, góc nghiêng thân, vận tốc rơi.
   - Microphone: Cường độ âm thanh dB (tiếng va đập mạnh, tiếng kêu cứu).
   - Cảm biến nhiệt: Thân nhiệt hồng ngoại (°C).
   - Vòng đeo tay BLE: Nhịp tim thời gian thực (BPM).
3. TỰ ĐỘNG GHI LOG VĂN BẢN (System Event Log) có mốc thời gian vào file `system_logs.txt`.
   Log này là nguồn tri thức thời gian thực phục vụ trực tiếp cho RAG Engine ở Bước 3.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import joblib

# Đảm bảo tương thích UTF-8 cho console trên Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import hàm train từ cùng package
try:
    from src.sensor_ml.train_model import train_sensor_model, FEATURE_COLUMNS, MODEL_PATH, DATA_FILE
except ImportError:
    from train_model import train_sensor_model, FEATURE_COLUMNS, MODEL_PATH, DATA_FILE

# Đường dẫn file nhật ký hệ thống
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SYSTEM_LOGS_PATH = os.path.join(PROJECT_ROOT, "system_logs.txt")


class SensorClassifier:
    """
    Lớp điều phối Phân loại Cảm biến & Ghi Log Tự Động.
    Thiết kế Singleton/Tối ưu bộ nhớ cho Edge AI.
    """

    def __init__(self, model_path: str = MODEL_PATH, log_file: str = SYSTEM_LOGS_PATH):
        self.model_path = model_path
        self.log_file = log_file
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.classes = []
        self._load_or_train_model()

    def _load_or_train_model(self):
        """
        Nạp mô hình đã huấn luyện từ đĩa.
        Nếu file chưa tồn tại (lần đầu chạy), tự động kích hoạt train_sensor_model.
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"[WARN] Chua tim thay file mo hinh tai {self.model_path}. Dang tien hanh huan luyen tu dong...", flush=True)
                train_sensor_model(data_path=DATA_FILE, save_path=self.model_path)

            print(f"[LOAD] Dang nap mo hinh Scikit-learn tu: {self.model_path}", flush=True)
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
            self.classes = payload.get("classes", list(self.model.classes_))
            print(f"[OK] Nap mo hinh thanh cong! Ho tro cac lop su kien: {self.classes}", flush=True)

        except Exception as e:
            print(f"[ERROR NAP MO HINH]: {str(e)}", file=sys.stderr, flush=True)
            raise e

    def classify_reading(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nhận vào 1 mẫu đọc cảm biến, dự đoán nhãn bằng Random Forest và phân tích ngữ nghĩa.
        
        Input reading dict gồm:
        - aspect_ratio: float
        - torso_angle: float
        - vertical_velocity: float
        - sound_db: float
        - body_temp: float
        - heart_rate: int/float
        """
        try:
            # 1. Chuẩn bị vector đặc trưng theo đúng thứ tự
            feature_vector = [float(reading.get(col, 0.0)) for col in self.feature_columns]
            df_input = pd.DataFrame([feature_vector], columns=self.feature_columns)

            # 2. Suy luận bằng Random Forest (< 1ms)
            pred_class = self.model.predict(df_input)[0]
            pred_probs = self.model.predict_proba(df_input)[0]
            confidence = float(max(pred_probs))

            # 3. Phân tầng mức độ nghiêm trọng (Log Level & Sensor Fusion Context)
            aspect_ratio = float(reading.get("aspect_ratio", 0.0))
            torso_angle = float(reading.get("torso_angle", 0.0))
            velocity = float(reading.get("vertical_velocity", 0.0))
            sound_db = float(reading.get("sound_db", 0.0))
            temp = float(reading.get("body_temp", 36.5))
            hr = int(reading.get("heart_rate", 75))

            if pred_class == "Fall":
                log_level = "CRITICAL"
                event_type = "FALL_DETECTED"
                description = (
                    f"PHÁT HIỆN TÉ NGÃ KHẨN CẤP: Góc nghiêng {torso_angle}°, tỉ lệ thân {aspect_ratio:.2f}, "
                    f"vận tốc rơi {velocity:.2f}m/s kèm âm thanh va đập mạnh {sound_db:.1f}dB. "
                    f"Nhịp tim người bệnh tăng cao ({hr} BPM). Cần hỗ trợ y tế ngay lập tức!"
                )
            elif pred_class == "Fever":
                log_level = "WARNING"
                event_type = "FEVER_DETECTED"
                description = (
                    f"CẢNH BÁO THÂN NHIỆT CAO: Đo được nhiệt độ {temp:.1f}°C (vượt ngưỡng 38.0°C), "
                    f"nhịp tim phản ứng {hr} BPM. Cần kiểm tra dấu hiệu mất nước và nguy cơ sốt nhiễm khuẩn."
                )
            elif pred_class == "Cardiac_Alert":
                log_level = "CRITICAL"
                event_type = "CARDIAC_ALERT"
                description = (
                    f"CẢNH BÁO TIM MẠCH BẤT THƯỜNG: Nhịp tim đo được {hr} BPM (vượt ngưỡng an toàn), "
                    f"tư thế góc nghiêng {torso_angle}°. Nghi ngờ cơn hồi hộp, nhịp nhanh hoặc đau ngực cấp."
                )
            else:
                log_level = "INFO"
                event_type = "NORMAL_ACTIVITY"
                description = (
                    f"TRẠNG THÁI BÌNH THƯỜNG: Thân nhiệt {temp:.1f}°C, nhịp tim {hr} BPM, "
                    f"tư thế sinh hoạt ổn định (góc nghiêng {torso_angle}°)."
                )

            return {
                "predicted_label": pred_class,
                "confidence": confidence,
                "log_level": log_level,
                "event_type": event_type,
                "description": description,
                "metrics": {
                    "aspect_ratio": aspect_ratio,
                    "torso_angle": torso_angle,
                    "vertical_velocity": velocity,
                    "sound_db": sound_db,
                    "body_temp": temp,
                    "heart_rate": hr
                }
            }

        except Exception as e:
            print(f"❌ [LỖI PHÂN LOẠI CẢM BIẾN]: {str(e)}", file=sys.stderr)
            return {
                "predicted_label": "Unknown",
                "confidence": 0.0,
                "log_level": "ERROR",
                "event_type": "SENSOR_ERROR",
                "description": f"Lỗi trong quá trình xử lý cảm biến: {str(e)}",
                "metrics": {}
            }

    def log_event(self, timestamp: str, level: str, event_type: str, 
                  metrics: Dict[str, Any], description: str) -> str:
        """
        TỰ ĐỘNG GHI 1 DÒNG NHẬT KÝ ĐỊNH DẠNG VĂN BẢN VÀO FILE system_logs.txt.
        Định dạng:
        [YYYY-MM-DD HH:MM:SS] [LEVEL] [EVENT_TYPE] [METRICS: ...] -> [DESCRIPTION]
        """
        try:
            metrics_str = (
                f"aspect_ratio={metrics.get('aspect_ratio', 0):.2f}, "
                f"angle={metrics.get('torso_angle', 0):.1f}°, "
                f"velocity={metrics.get('vertical_velocity', 0):.2f}m/s, "
                f"sound={metrics.get('sound_db', 0):.1f}dB, "
                f"temp={metrics.get('body_temp', 0):.1f}°C, "
                f"hr={metrics.get('heart_rate', 0)}BPM"
            )
            log_line = f"[{timestamp}] [{level}] [{event_type}] [METRICS: {metrics_str}] -> {description}\n"

            # Đảm bảo thư mục cha tồn tại
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

            # Ghi nối tiếp (append mode) an toàn
            with open(self.log_file, mode="a", encoding="utf-8") as f:
                f.write(log_line)

            return log_line.strip()

        except Exception as e:
            print(f"❌ [LỖI GHI LOG VÀO FILE]: {str(e)}", file=sys.stderr)
            return ""

    def process_and_log(self, reading: Dict[str, Any], timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Quy trình trọn gói: Tiếp nhận mẫu cảm biến -> Phân loại ML -> Tự động ghi Log vào system_logs.txt.
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = self.classify_reading(reading)
        log_line = self.log_event(
            timestamp=timestamp,
            level=result["log_level"],
            event_type=result["event_type"],
            metrics=result["metrics"],
            description=result["description"]
        )
        result["timestamp"] = timestamp
        result["log_line"] = log_line
        return result


def run_sensor_simulation(mock_file: str = DATA_FILE, delay: float = 0.05, max_records: int = 50):
    """
    Hàm mô phỏng luồng cảm biến thực tế đọc từ file `mock_sensor_data.txt`,
    thực hiện phân loại qua Random Forest và tự động ghi nhật ký hệ thống.
    """
    print("=" * 80, flush=True)
    print("[SENSOR STREAM SIMULATOR] BAT DAU MO PHONG DOC CAM BIEN & TU DONG GHI LOG", flush=True)
    print("=" * 80, flush=True)

    try:
        classifier = SensorClassifier()

        if not os.path.exists(mock_file):
            raise FileNotFoundError(f"Khong tim thay file: {mock_file}")

        df = pd.read_csv(mock_file, comment="#")
        print(f"[STREAM] Tim thay {len(df)} ban ghi cam bien can mo phong.", flush=True)
        print("-" * 80, flush=True)

        processed_count = 0
        fall_count = 0
        fever_count = 0
        cardiac_count = 0

        for idx, row in df.iterrows():
            if processed_count >= max_records:
                break

            timestamp = str(row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            reading = {
                "aspect_ratio": row["aspect_ratio"],
                "torso_angle": row["torso_angle"],
                "vertical_velocity": row["vertical_velocity"],
                "sound_db": row["sound_db"],
                "body_temp": row["body_temp"],
                "heart_rate": row["heart_rate"]
            }

            # Phân loại và Tự động ghi Log
            res = classifier.process_and_log(reading, timestamp=timestamp)

            # Đếm thống kê
            label = res["predicted_label"]
            if label == "Fall":
                fall_count += 1
                icon = "[CRITICAL_FALL]"
            elif label == "Fever":
                fever_count += 1
                icon = "[WARNING_FEVER]"
            elif label == "Cardiac_Alert":
                cardiac_count += 1
                icon = "[CARDIAC_ALERT]"
            else:
                icon = "[NORMAL_INFO ]"

            print(f"{icon} [{timestamp}] Pred: {label:<13} | Conf: {res['confidence']*100:.1f}% | {res['description'][:60]}...", flush=True)

            processed_count += 1
            if delay > 0:
                time.sleep(delay)

        print("-" * 80, flush=True)
        print("[KET QUA MO PHONG LUONG CAM BIEN]:", flush=True)
        print(f"  - Tong so su kien da xu ly : {processed_count}", flush=True)
        print(f"  - So ca phat hien te nga   : {fall_count} (Da ghi Log khan cap)", flush=True)
        print(f"  - So ca phat hien sot cao  : {fever_count} (Da ghi Log canh bao)", flush=True)
        print(f"  - So ca tim mach bat thuong: {cardiac_count} (Da ghi Log canh bao)", flush=True)
        print(f"[LOG FILE] Kiem tra file nhat ky tai: {classifier.log_file}", flush=True)
        print("=" * 80, flush=True)

    except Exception as e:
        print(f"[ERROR MO PHONG LUONG]: {str(e)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run_sensor_simulation(delay=0.01)
