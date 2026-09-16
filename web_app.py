# -*- coding: utf-8 -*-
"""
HỆ THỐNG AI GIÁM SÁT NGƯỜI CAO TUỔI (LOCAL EDGE AI DASHBOARD)
Phiên bản: 3.0 Clean Modular Architecture
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems
"""

import os
import sys
from datetime import datetime
import requests
import streamlit as st
import pandas as pd

# Đảm bảo UTF-8 cho console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.config import API_BASE_URL, SYSTEM_LOGS_PATH, DATA_FILE
from src.rag.rag_engine import get_rag_engine
from src.sensor_ml.sensor_classifier import SensorClassifier
from src.voice.voice_service import get_voice_service
from src.ui.styles import apply_custom_styles
from src.ui.components import (
    render_header_and_vitals,
    render_chat_interface,
    render_logs_viewer,
    render_sidebar
)

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT & ÁP DỤNG GIAO DIỆN THEME
# ==============================================================================
st.set_page_config(
    page_title="AuraCare AI • Giám sát Người cao tuổi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles(st)

# ==============================================================================
# KHỞI TẠO SINGLETON ENGINES
# ==============================================================================
@st.cache_resource
def load_core_engines():
    """Khởi tạo các module máy học, RAG và Giọng nói cục bộ."""
    return get_rag_engine(), SensorClassifier(), get_voice_service()

rag_engine, sensor_classifier, voice_service = load_core_engines()


def query_ai_service(question: str) -> dict:
    """Gửi câu hỏi tới FastAPI server nếu online, hoặc gọi trực tiếp RAG Engine cục bộ."""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/chat", json={"message": question}, timeout=25)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Direct Engine Fallback
    res = rag_engine.query(question)
    return {
        "status": "success",
        "question": question,
        "answer": res["answer"],
        "sources": res.get("sources", []),
        "engine": res.get("engine", "AuraCare"),
        "tools_used": res.get("tools_used", []),
        "disclaimer": res.get("disclaimer"),
        "crag_status": res.get("crag_status"),
        "graph_entities": res.get("graph_entities", []),
        "expanded_queries": res.get("expanded_queries", []),
        "timestamp": res.get("timestamp")
    }


def get_latest_system_logs(limit: int = 50, level_filter: str = "ALL") -> list:
    """Lấy danh sách nhật ký từ API hoặc phân tích trực tiếp file system_logs.txt."""
    try:
        params = {"limit": limit}
        if level_filter and level_filter != "ALL":
            params["level"] = level_filter
        resp = requests.get(f"{API_BASE_URL}/api/logs", params=params, timeout=3)
        if resp.status_code == 200:
            return resp.json().get("logs", [])
    except Exception:
        pass

    logs = []
    logs_file_str = str(SYSTEM_LOGS_PATH)
    if os.path.exists(logs_file_str):
        with open(logs_file_str, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        lines.reverse()

        for line in lines[:limit]:
            lvl = "INFO"
            if "CRITICAL" in line: lvl = "CRITICAL"
            elif "WARNING" in line: lvl = "WARNING"

            if level_filter != "ALL" and lvl != level_filter:
                continue

            logs.append({
                "timestamp": line.split("]")[0].replace("[", "") if "]" in line else "",
                "level": lvl,
                "description": line,
                "raw_log": line
            })
    return logs


def simulate_random_telemetry():
    """Đọc 1 dòng mẫu ngẫu nhiên từ mock_sensor_data.txt để mô phỏng sự kiện phần cứng."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, comment="#")
        sample_row = df.sample(n=1).iloc[0]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sample = {
            "aspect_ratio": float(sample_row["aspect_ratio"]),
            "torso_angle": float(sample_row["torso_angle"]),
            "vertical_velocity": float(sample_row["vertical_velocity"]),
            "sound_db": float(sample_row["sound_db"]),
            "body_temp": float(sample_row["body_temp"]),
            "heart_rate": int(sample_row["heart_rate"])
        }
        res = sensor_classifier.process_and_log(sample, timestamp=now_str)
        rag_engine._load_local_documents_cache()
        return res
    return None


# ==============================================================================
# KHỞI TẠO SESSION STATE
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Chào bạn, tôi là **Trợ lý AI Giám sát Người cao tuổi AuraCare**.\n\n"
                "Hệ thống đang theo dõi đa cảm biến (Camera, Micro, Thân nhiệt, Nhịp tim). "
                "Bạn có thể gõ câu hỏi hoặc **nhấn biểu tượng Micro** bên dưới để nói trực tiếp."
            ),
            "sources": ["elderly_first_aid.txt", "system_logs.txt"],
            "engine": "AuraCare Edge Engine"
        }
    ]

# ==============================================================================
# VẼ TOÀN BỘ CÁC MODULE GIAO DIỆN
# ==============================================================================
# 1. Sidebar điều khiển & CSDL hồ sơ
render_sidebar(st)

# 2. Header thương hiệu & Lưới 4 thẻ sinh hiệu
recent_logs = get_latest_system_logs(limit=1)
render_header_and_vitals(st, recent_logs)

# 3. Thân ứng dụng chia làm 2 cột: Chatbot AI và Nhật ký cảm biến
col_chat, col_logs = st.columns([1.15, 0.85], gap="medium")

with col_chat:
    render_chat_interface(st, query_ai_service, voice_service)

with col_logs:
    render_logs_viewer(st, get_latest_system_logs, simulate_random_telemetry)

# 4. Footer hệ thống
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; font-size: 12px; margin-top: 10px;">
    AuraCare Edge AI • Hệ thống Giám sát Người cao tuổi Nội bộ • Bảo mật Cục bộ 100% • 
    Kiến trúc: Clean Modular Architecture • Trạng thái: Ổn định
</div>
""", unsafe_allow_html=True)
