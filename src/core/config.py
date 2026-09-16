# -*- coding: utf-8 -*-
"""
HỆ THỐNG CẤU HÌNH TRUNG TÂM (CORE SYSTEM CONFIGURATION)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare (Local Edge AI)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# ------------------------------------------------------------------------------
# ĐƯỜNG DẪN THƯ MỤC CƠ BẢN
# ------------------------------------------------------------------------------
CORE_DIR = Path(__file__).resolve().parent
SRC_DIR = CORE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
AUDIO_DIR = DATA_DIR / "audio"
MEDICAL_DOCS_DIR = DATA_DIR / "medical_docs"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# Đường dẫn các file dữ liệu then chốt
DB_PATH = DATA_DIR / "elderly_care.db"
SYSTEM_LOGS_PATH = PROJECT_ROOT / "system_logs.txt"

# Dữ liệu cảm biến mô phỏng (hỗ trợ cả đường dẫn trong data/ và root)
MOCK_SENSOR_DATA_PATH = DATA_DIR / "mock_sensor_data.txt"
if not MOCK_SENSOR_DATA_PATH.exists() and (PROJECT_ROOT / "mock_sensor_data.txt").exists():
    MOCK_SENSOR_DATA_PATH = PROJECT_ROOT / "mock_sensor_data.txt"

DATA_FILE = str(MOCK_SENSOR_DATA_PATH)
SENSOR_MODEL_PATH = MODELS_DIR / "sensor_rf_model.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"

# ------------------------------------------------------------------------------
# CẤU HÌNH MÁY CHỦ API VÀ GIAO DIỆN
# ------------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", 8001))
API_BASE_URL = os.environ.get("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", 8501))

# ------------------------------------------------------------------------------
# CẤU HÌNH LLM VÀ EMBEDDINGS (EDGE HYBRID)
# ------------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2:7b")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Cloud/External LLM fallback (nếu cấu hình trong .env)
DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://api.xkiro.com/v1"

LLM_API_KEY = os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

CANDIDATE_MODELS: List[str] = [
    "mistralai/ministral-8b",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-medium-3.5",
    "mistralai/mistral-small-2603",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-v3.2"
]

# ------------------------------------------------------------------------------
# CẤU HÌNH NGƯỠNG AN TOÀN SINH HIỆU LÂM SÀNG
# ------------------------------------------------------------------------------
VITALS_THRESHOLDS: Dict[str, Any] = {
    "heart_rate": {
        "min_normal": 60,
        "max_normal": 100,
        "warning_high": 110,
        "critical_high": 120,
        "critical_low": 50,
        "unit": "BPM"
    },
    "body_temp": {
        "min_normal": 36.2,
        "max_normal": 37.4,
        "warning_fever": 37.8,
        "critical_fever": 38.5,
        "unit": "°C"
    },
    "blood_pressure": {
        "systolic_max_normal": 139,
        "systolic_crisis": 180,
        "diastolic_crisis": 120,
        "unit": "mmHg"
    },
    "fall_detection": {
        "torso_angle_fall_threshold": 60.0,    # Góc nghiêng thân người > 60°
        "vertical_velocity_impact": 1.5,        # Tốc độ rơi thẳng đứng
        "sound_impact_db": 80.0                 # Âm thanh va đập lớn
    }
}

# ------------------------------------------------------------------------------
# THÔNG TIN BỆNH NHÂN & ĐỘI NGŨ Y TẾ MẶC ĐỊNH
# ------------------------------------------------------------------------------
PATIENT_DEFAULT_INFO: Dict[str, Any] = {
    "full_name": "Cụ Nguyễn Văn An",
    "age": 82,
    "patient_code": "#ELD-8402",
    "room": "Phòng 102 - Tầng 1",
    "medical_history": "Tăng huyết áp độ 2, Tiểu đường Type 2, Đau thoái hóa khớp gối, từng té ngã năm 2024",
    "allergies": "Dị ứng tuyệt đối kháng sinh nhóm Penicillin (Amoxicillin, Augmentin, Ampicillin)",
    "doctor_name": "BS. CKI Trần Minh Tuấn",
    "doctor_phone": "0912.345.678",
    "emergency_phone": "115"
}

PRIMARY_CAREGIVER_DEFAULT: Dict[str, Any] = {
    "full_name": "Nguyễn Hữu Nghĩa",
    "role": "Người bảo hộ chính",
    "relationship": "Con trai trưởng",
    "phone": "0908.123.456"
}
