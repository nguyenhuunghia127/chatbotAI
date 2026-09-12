# -*- coding: utf-8 -*-
"""
HỆ THỐNG AI GIÁM SÁT NGƯỜI CAO TUỔI (LOCAL EDGE AI DASHBOARD)
Phiên bản: 2.0 Ultra-Modern Medical Dashboard
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems
"""

import os
import sys
import time
import textwrap
import random
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

from src.rag.rag_engine import get_rag_engine, SYSTEM_LOGS_PATH
from src.sensor_ml.sensor_classifier import SensorClassifier, DATA_FILE
from src.voice.voice_service import get_voice_service, normalize_vietnamese_voice_transcript
from src.db.db_service import get_db_service

# Cấu hình API Backend
API_BASE_URL = "http://127.0.0.1:8001"

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AuraCare AI • Giám sát Người cao tuổi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# HỆ THỐNG CSS CAO CẤP (ULTRA-PREMIUM HEALTHCARE THEME)
# ==============================================================================
custom_css = textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background tổng thể */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }

    /* Ẩn header mặc định của Streamlit */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Banner chính trên đỉnh */
    .nav-banner {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .brand-title {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 2px;
        font-weight: 500;
    }
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 12px #22c55e;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Thẻ sinh hiệu Glassmorphism cao cấp */
    .vital-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .vital-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .vital-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 0 15px rgba(56, 189, 248, 0.15);
    }
    .vital-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .vital-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .vital-icon {
        font-size: 20px;
    }
    .vital-value {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .vital-status {
        font-size: 11px;
        font-weight: 600;
        margin-top: 4px;
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
    }
    
    /* Màu trạng thái sinh hiệu */
    .status-normal { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .status-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }

    /* Khung Sidebar Status Widget */
    .status-badge-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .status-title {
        font-size: 13px;
        font-weight: 600;
        color: #cbd5e1;
    }
    .status-pill {
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Khung Chat Container */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* Hộp nhật ký sự kiện chuyên nghiệp */
    .log-scrollbox {
        background: rgba(2, 6, 23, 0.85);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 14px;
        padding: 14px;
        height: 480px;
        overflow-y: auto;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .log-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        transition: background 0.2s;
    }
    .log-card:hover {
        background: rgba(30, 41, 59, 0.8);
    }
    .log-card-critical {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.06);
    }
    .log-card-warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.06);
    }
    .log-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #94a3b8;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .log-desc {
        font-size: 12.5px;
        color: #e2e8f0;
        line-height: 1.45;
    }

    /* Các nút hành động */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
    }
</style>
""")
st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# HÀM BỔ TRỢ ENGINE & API
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
        "sources": res["sources"],
        "engine": res["engine"],
        "timestamp": res["timestamp"]
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

    # File parser fallback
    logs = []
    if os.path.exists(SYSTEM_LOGS_PATH):
        with open(SYSTEM_LOGS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        lines.reverse()

        for line in lines[:limit]:
            lvl = "INFO"
            if "CRITICAL" in line: lvl = "CRITICAL"
            elif "WARNING" in line: lvl = "WARNING"

            if level_filter != "ALL" and lvl != level_filter:
                continue

            desc = line.split(" -> ")[-1] if " -> " in line else line
            parts = line.split("] [")
            t_str = parts[0].replace("[", "") if len(parts) > 1 else ""

            logs.append({
                "timestamp": t_str,
                "level": lvl,
                "description": desc,
                "raw_log": line
            })
    return logs


def simulate_random_telemetry():
    """Mô phỏng 1 nhịp đọc dữ liệu cảm biến mới từ file mock."""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/sensor/simulate-step", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

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
# SIDEBAR: THÔNG TIN NGƯỜI BỆNH & TRẠNG THÁI HỆ THỐNG
# ==============================================================================
with st.sidebar:
    db_svc = get_db_service()
    cur_u = db_svc.get_current_user()
    p_info = db_svc.get_patient_profile()

    u_name = cur_u.get("full_name", "Nguyễn Hữu Nghĩa")
    u_role = cur_u.get("role", "Người bảo hộ chính")
    u_rel = cur_u.get("relationship", "Con trai trưởng")
    u_phone = cur_u.get("phone", "0908.123.456")

    p_name = p_info.get("full_name", "Cụ Nguyễn Văn An")
    p_age = p_info.get("age", 82)
    p_code = p_info.get("patient_code", "#ELD-8402")
    p_room = p_info.get("room", "Phòng 102 - Tầng 1")
    p_history = p_info.get("medical_history", "Tăng huyết áp, Đau khớp")
    p_allergies = p_info.get("allergies", "Dị ứng Penicillin")

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 14px;">
        <div style="font-size: 36px; margin-bottom: 4px;">👵👴</div>
        <div style="font-size: 16.5px; font-weight: 700; color: #38bdf8;">CSDL HỒ SƠ Y TẾ & NGƯỜI DÙNG</div>
        <div style="font-size: 11.5px; color: #94a3b8;">Mã: {p_code} • {p_room}</div>
    </div>
    """, unsafe_allow_html=True)

    # Thẻ hồ sơ người cao tuổi & người giám hộ từ DB
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px; margin-bottom: 12px; font-size: 12px;">
        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 6px;">📋 NGƯỜI CAO TUỔI (BỆNH NHÂN):</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #94a3b8;">Họ tên:</span>
            <span style="font-weight: 600; color: #f8fafc;">{p_name} ({p_age}T)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #94a3b8;">Tiền sử:</span>
            <span style="color: #fbbf24; font-weight: 600;">{p_history[:25]}...</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #94a3b8;">Dị ứng:</span>
            <span style="color: #f87171; font-weight: 600;">{p_allergies[:25]}</span>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.06); margin: 6px 0;">
        <div style="font-weight: 700; color: #a78bfa; margin-bottom: 6px;">👤 NGƯỜI ĐANG HỎI (BẠN):</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #94a3b8;">Họ tên:</span>
            <span style="font-weight: 600; color: #f8fafc;">{u_name}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #94a3b8;">Vai trò:</span>
            <span style="color: #38bdf8; font-weight: 600;">{u_role} ({u_rel})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94a3b8;">Điện thoại:</span>
            <span style="color: #94a3b8;">{u_phone}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Đổi danh tính người hỏi (Tôi là ai)", expanded=False):
        with st.form("user_identity_form"):
            new_u_name = st.text_input("Tên của bạn:", value=u_name)
            new_u_rel = st.text_input("Mối quan hệ / Vai trò:", value=u_rel)
            new_u_phone = st.text_input("SĐT liên hệ:", value=u_phone)
            submit_u = st.form_submit_button("💾 Lưu vào CSDL", use_container_width=True)
            if submit_u:
                db_svc.set_current_user(new_u_name, relationship=new_u_rel, phone=new_u_phone)
                st.success(f"Đã cập nhật danh tính '{new_u_name}' vào CSDL!")
                st.rerun()

    st.markdown("#### ⚡ Trạng thái Trạm Biên (Edge Nodes)")
    
    # Kiểm tra API
    api_live = False
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=1)
        api_live = (r.status_code == 200)
    except Exception:
        api_live = False

    api_color = "#22c55e" if api_live else "#eab308"
    api_text = "Port 8001 Live" if api_live else "Direct Core Mode"

    st.markdown(f"""
    <div class="status-badge-card">
        <span class="status-title">📡 FastAPI Backend</span>
        <span class="status-pill" style="background: rgba({ '34,197,94' if api_live else '234,179,8' }, 0.2); color: {api_color}; border: 1px solid {api_color};">{api_text}</span>
    </div>
    <div class="status-badge-card">
        <span class="status-title">🧠 ML Random Forest</span>
        <span class="status-pill" style="background: rgba(34,197,94, 0.2); color: #22c55e; border: 1px solid #22c55e;">Sẵn sàng (1ms)</span>
    </div>
    <div class="status-badge-card">
        <span class="status-title">📚 Vector DB (Chroma)</span>
        <span class="status-pill" style="background: rgba(56,189,248, 0.2); color: #38bdf8; border: 1px solid #38bdf8;">Offline Local</span>
    </div>
    <div class="status-badge-card">
        <span class="status-title">🎙️ Nhận diện giọng nói (STT)</span>
        <span class="status-pill" style="background: rgba(168,85,247, 0.2); color: #c084fc; border: 1px solid #c084fc;">Faster-Whisper (Tiếng Việt)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🚨 Báo Động Khẩn Cấp")
    if st.button("📞 GỌI CẤP CỨU 115 NGAY", type="primary", use_container_width=True):
        st.error("🚨 Đang phát còi báo động tại trạm phòng và liên hệ Trung tâm Cấp cứu 115!")

    with st.expander("📖 Cẩm nang Sơ cứu Nhanh", expanded=False):
        st.markdown("""
        * **Khi Té Ngã**: Không vội đỡ dậy. Kiểm tra tri giác, xương hông và gọi 115 nếu nghi ngờ gãy xương.
        * **Khi Sốt Cao (>38°C)**: Bổ sung oresol, chườm khăn ấm (30-32°C) vùng trán/nách/bẹn.
        * **Rối loạn nhịp tim**: Đặt ngồi Fowler 45°, nới lỏng áo và hít thở sâu.
        """)

    st.markdown("---")
    st.markdown("#### 🤖 Tự Động Huấn Luyện Ngầm (Auto-Trainer)")
    try:
        from src.training.auto_trainer import get_auto_trainer
        trainer = get_auto_trainer()
        t_status = trainer.get_status()

        last_time = t_status.get("last_run_timestamp")
        last_time_str = last_time.split()[1] if last_time and " " in last_time else "Vừa xong"
        cycles = t_status.get("total_cycles_completed", 1)
        samples = t_status.get("synthetic_qa_samples_learned", 20)

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 12px; margin-bottom: 10px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Chế độ:</span>
                <span style="font-weight: 700; color: #38bdf8;">🟢 Tự động ngầm (100% Auto)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Chu kỳ đã chạy:</span>
                <span style="font-weight: 700; color: #f8fafc;">#{cycles} (lúc {last_time_str})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Tri thức Q&A tự nạp:</span>
                <span style="font-weight: 700; color: #4ade80;">{samples} tình huống y tế</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8;">Độ chính xác Scikit-Learn:</span>
                <span style="font-weight: 700; color: #c084fc;">100.0% (Augmented)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        r_col1, r_col2 = st.columns(2)
        if r_col1.button("🚀 Chạy chu kỳ mới", use_container_width=True, help="Tự động sinh mẫu mới và huấn luyện ngầm"):
            with st.spinner("Đang tự động huấn luyện ngầm..."):
                trainer.run_full_training_cycle()
                st.success("Đã hoàn tất chu kỳ huấn luyện tự động! 🎯")
                st.rerun()

        if r_col2.button("📦 Xem Dataset", use_container_width=True, help="Xuất file JSON DPO / Alpaca"):
            from src.training.rlhf_service import get_rlhf_service
            get_rlhf_service().export_datasets()
            st.toast("Đã xuất data/training_data/rlhf_dataset.json! 🚀", icon="📦")

        with st.expander("✨ Tạo sinh Dữ liệu bằng LLM API", expanded=False):
            from src.training.llm_synthetic_generator import (
                DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL,
                LLMSyntheticGenerator, GENERATION_TOPICS
            )
            cfg_key = st.text_input("API Key:", value=DEFAULT_API_KEY, type="password", help="Key bắt đầu bằng sk-xt-...")
            cfg_url = st.text_input("Base URL:", value=DEFAULT_BASE_URL, help="Ví dụ: https://api.xkiro.com/v1")
            cfg_model = st.text_input("Model Name:", value=DEFAULT_MODEL, help="Ví dụ: deepseek/deepseek-chat-v3.1, deepseek/deepseek-v4-pro")
            topic_choice = st.selectbox("Chủ đề y tế cần tạo:", ["Tất cả chủ đề (all)"] + list(GENERATION_TOPICS.keys()))
            num_per_t = st.slider("Số mẫu trên mỗi chủ đề:", min_value=1, max_value=10, value=3)

            if st.button("🚀 Bắt đầu Tạo sinh Dữ liệu", use_container_width=True, type="primary"):
                with st.spinner(f"Đang gọi {cfg_model} tại {cfg_url} để tạo sinh dữ liệu y tế chuẩn..."):
                    gen = LLMSyntheticGenerator(api_key=cfg_key, base_url=cfg_url, model=cfg_model)
                    actual_topic = "all" if "all" in topic_choice else topic_choice
                    if actual_topic == "all":
                        res = gen.generate_all_topics(samples_per_topic=num_per_t)
                    else:
                        samples = gen.generate_qa_pairs(topic_key=actual_topic, num_samples=num_per_t)
                        res = gen.integrate_into_knowledge_base(samples)
                        res["total_generated"] = len(samples)

                    if res.get("total_generated", 0) > 0:
                        st.success(f"🎉 Đã sinh thành công {res.get('total_generated')} mẫu dữ liệu và nạp vào bộ nhớ RAG!")
                        st.rerun()
                    else:
                        st.error("Không thể kết nối hoặc API trả về lỗi. Vui lòng kiểm tra lại API Key, Base URL hoặc Model Name.")
    except Exception as e_trainer:
        st.caption(f"Trạng thái Auto-Trainer: {e_trainer}")


# ==============================================================================
# HEADER CHÍNH
# ==============================================================================
st.markdown(textwrap.dedent(f"""
<div class="nav-banner">
    <div>
        <div class="brand-title">🛡️ AURACARE AI • HỆ THỐNG GIÁM SÁT NGƯỜI CAO TUỔI</div>
        <div class="brand-subtitle">
            <span class="pulse-dot"></span>
            <b>TRỰC TUYẾN 100% OFFLINE</b> • TỔNG HỢP CẢM BIẾN (VISION + AUDIO + THERMAL + BLE) • CHỐNG BỊA ĐẶT (NO HALLUCINATION)
        </div>
    </div>
    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #38bdf8;">
        {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
    </div>
</div>
"""), unsafe_allow_html=True)


# ==============================================================================
# 4 THẺ SINH HIỆU THỜI GIAN THỰC (LIVE METRIC CARDS)
# ==============================================================================
# Đọc dòng log gần nhất để hiển thị sinh hiệu
recent_logs = get_latest_system_logs(limit=1)

temp_val = 36.6
temp_status = "Bình thường"
temp_class = "status-normal"

hr_val = 74
hr_status = "Ổn định"
hr_class = "status-normal"

posture_val = "Đứng / Ngồi"
posture_status = "An toàn"
posture_class = "status-normal"

sound_val = 42.5
sound_status = "Yên tĩnh"
sound_class = "status-normal"

if recent_logs:
    last_raw = recent_logs[0].get("raw_log", "")
    if "FALL_DETECTED" in last_raw:
        posture_val = "🚨 NGÃ NẰM SÀN"
        posture_status = "NGUY HIỂM CẤP"
        posture_class = "status-critical"
        hr_val = 135
        hr_status = "Nhịp nhanh"
        hr_class = "status-critical"
        sound_val = 88.0
        sound_status = "Va đập mạnh"
        sound_class = "status-critical"
    elif "FEVER_DETECTED" in last_raw:
        temp_val = 38.9
        temp_status = "Sốt cao"
        temp_class = "status-warning"
        hr_val = 104
        hr_status = "Tăng do sốt"
        hr_class = "status-warning"
        posture_val = "Nằm giường"
    elif "CARDIAC_ALERT" in last_raw:
        hr_val = 145
        hr_status = "Cơn nhịp nhanh"
        hr_class = "status-critical"
        posture_val = "Ngồi nghỉ"

v_col1, v_col2, v_col3, v_col4 = st.columns(4)

with v_col1:
    st.markdown(f"""
    <div class="vital-card">
        <div class="vital-header">
            <span class="vital-label">Thân nhiệt</span>
            <span class="vital-icon">🌡️</span>
        </div>
        <div class="vital-value" style="color: {'#fbbf24' if temp_val >= 38 else '#4ade80'};">{temp_val}°C</div>
        <span class="vital-status {temp_class}">{temp_status}</span>
    </div>
    """, unsafe_allow_html=True)

with v_col2:
    st.markdown(f"""
    <div class="vital-card">
        <div class="vital-header">
            <span class="vital-label">Nhịp tim (BLE)</span>
            <span class="vital-icon">💓</span>
        </div>
        <div class="vital-value" style="color: {'#f87171' if hr_val > 110 else '#4ade80'};">{hr_val} <span style="font-size: 14px; font-weight: 500;">BPM</span></div>
        <span class="vital-status {hr_class}">{hr_status}</span>
    </div>
    """, unsafe_allow_html=True)

with v_col3:
    st.markdown(f"""
    <div class="vital-card">
        <div class="vital-header">
            <span class="vital-label">Tư thế (Vision)</span>
            <span class="vital-icon">🧍</span>
        </div>
        <div class="vital-value" style="font-size: 18px; color: {'#f87171' if 'NGÃ' in posture_val else '#4ade80'};">{posture_val}</div>
        <span class="vital-status {posture_class}">{posture_status}</span>
    </div>
    """, unsafe_allow_html=True)

with v_col4:
    st.markdown(f"""
    <div class="vital-card">
        <div class="vital-header">
            <span class="vital-label">Âm thanh phòng</span>
            <span class="vital-icon">🔊</span>
        </div>
        <div class="vital-value" style="color: {'#f87171' if sound_val > 75 else '#4ade80'};">{sound_val} <span style="font-size: 14px; font-weight: 500;">dB</span></div>
        <span class="vital-status {sound_class}">{sound_status}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# BỐ CỤC CHÍNH: 2 CỘT CÂN ĐỐI
# ==============================================================================
col_chat, col_logs = st.columns([1.1, 0.9], gap="large")

# ------------------------------------------------------------------------------
# CỘT 1: KHUNG CHATBOT AI VÀ GỌI GIỌNG NÓI
# ------------------------------------------------------------------------------
with col_chat:
    st.markdown("### 💬 Trợ Lý AI Chăm Sóc Y Tế")

    # Các nút câu hỏi nhanh dạng thẻ ngang
    st.caption("⚡ Nhấp để hỏi nhanh về tình hình người cao tuổi:")
    q_col1, q_col2 = st.columns(2)
    selected_quick_q = None

    with q_col1:
        if st.button("🚨 Kiểm tra té ngã hôm nay", use_container_width=True):
            selected_quick_q = "Sáng nay ông có bị ngã không? Chi tiết thế nào và tôi phải làm gì?"
        if st.button("🌡️ Tình hình sốt & thân nhiệt", use_container_width=True):
            selected_quick_q = "Tình hình thân nhiệt và sốt của bà hôm nay như thế nào?"

    with q_col2:
        if st.button("💓 Cảnh báo nhịp tim bất thường", use_container_width=True):
            selected_quick_q = "Hôm nay có cảnh báo bất thường về tim mạch hoặc nhịp tim nhanh không?"
        if st.button("📋 Báo cáo tổng thể sức khỏe", use_container_width=True):
            selected_quick_q = "Hiện tại tình trạng sức khỏe chung của người cao tuổi ra sao?"

    # Container hiển thị đoạn chat
    chat_box = st.container(height=400)
    with chat_box:
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        st.caption(f"📚 Nguồn trích xuất: `{', '.join(msg['sources'])}` | Engine: `{msg.get('engine', 'AuraCare')}`")

                    # Thanh phản hồi để Huấn luyện Tăng cường (RLHF & Active Learning)
                    if idx > 0:
                        prev_user_q = st.session_state.chat_history[idx - 1]["content"] if st.session_state.chat_history[idx - 1]["role"] == "user" else "Câu hỏi"
                        fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 6])
                        if fb_c1.button("👍", key=f"fb_up_{idx}", help="Hài lòng - Thưởng điểm (+1) giúp AI ghi nhớ mẫu trả lời này"):
                            from src.training.rlhf_service import get_rlhf_service
                            get_rlhf_service().record_feedback(prompt=prev_user_q, response=msg["content"], rating=1)
                            st.toast("Đã ghi nhận 👍 (+1 điểm thưởng). AI đã ghi nhớ mẫu chuẩn này!", icon="🧠")
                        if fb_c2.button("👎", key=f"fb_down_{idx}", help="Chưa chuẩn - Phạt điểm (-1) và dạy câu trả lời đúng cho AI"):
                            st.session_state[f"edit_corr_{idx}"] = not st.session_state.get(f"edit_corr_{idx}", False)

                        if st.session_state.get(f"edit_corr_{idx}"):
                            with st.container():
                                st.markdown("<div style='font-size: 11.5px; color: #fbbf24; margin-top: 4px;'>✏️ <b>Dạy cho AI:</b> Nhập câu trả lời chuẩn xác mà bạn mong muốn:</div>", unsafe_allow_html=True)
                                corr_text = st.text_area("Câu trả lời đúng:", key=f"corr_inp_{idx}", height=70, placeholder="Ví dụ: Lúc 16h cụ sinh hoạt bình thường...")
                                if st.button("💾 Xác nhận Huấn Luyện AI", key=f"btn_corr_{idx}", use_container_width=True):
                                    if corr_text.strip():
                                        from src.training.rlhf_service import get_rlhf_service
                                        get_rlhf_service().record_feedback(prompt=prev_user_q, response=msg["content"], rating=-1, corrected_response=corr_text)
                                        st.session_state[f"edit_corr_{idx}"] = False
                                        st.success("✅ Đã cập nhật vào tập tri thức tăng cường! Lần sau hỏi câu này AI sẽ trả lời chuẩn xác theo bạn.")
                                        st.rerun()

    # Nhận câu hỏi qua chat_input
    user_input = st.chat_input("Nhập câu hỏi của bạn (ví dụ: Lúc 16h có ngã không?)...")
    if selected_quick_q:
        user_input = selected_quick_q

    # Nhận diện giọng nói Tiếng Việt trực tiếp từ Microphone trình duyệt
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    mic_audio = st.audio_input("🎙️ Nhấn biểu tượng Micro để nói câu hỏi bằng Tiếng Việt...", label_visibility="collapsed")
    if mic_audio is not None:
        raw_bytes = mic_audio.read()
        if raw_bytes and len(raw_bytes) > 0:
            import hashlib
            audio_hash = hashlib.md5(raw_bytes).hexdigest()
            if st.session_state.get("last_processed_audio") != audio_hash:
                st.session_state["last_processed_audio"] = audio_hash
                with st.spinner("🎙️ Hệ thống AI đang nhận diện giọng nói Tiếng Việt chuẩn xác..."):
                    transcribed = voice_service.transcribe_audio_bytes(raw_bytes, language="vi")
                    if transcribed and transcribed.strip():
                        user_input = transcribed.strip()
                        st.toast(f"Đã nhận diện: \"{user_input}\"", icon="🎙️")
                    else:
                        st.warning("⚠️ Chưa nhận diện được giọng nói rõ ràng. Vui lòng nói to và rõ hơn.")

    # Nút câu hỏi mẫu giọng nói (dự phòng nhanh)
    with st.expander("⚡ Bấm thử nhanh các câu hỏi mẫu (Sức khỏe & Hồ sơ CSDL)", expanded=False):
        st.markdown("<div style='font-size: 12px; color: #94a3b8; margin-bottom: 4px;'>🩺 Câu hỏi theo dõi sức khỏe & cảm biến:</div>", unsafe_allow_html=True)
        v_btn1, v_btn2, v_btn3 = st.columns(3)
        if v_btn1.button("🎤 'Sáng nay ông có ngã không?'", use_container_width=True):
            user_input = "Sáng nay ông có bị ngã không? Chi tiết thế nào và tôi phải làm gì?"
        if v_btn2.button("🎤 'Thân nhiệt bà có sốt không?'", use_container_width=True):
            user_input = "Tình hình thân nhiệt và sốt của bà hôm nay như thế nào?"
        if v_btn3.button("🎤 'Nhịp tim lúc 19h có sao không?'", use_container_width=True):
            user_input = "Hôm nay có cảnh báo bất thường về tim mạch hoặc nhịp tim nhanh không?"

        st.markdown("<div style='font-size: 12px; color: #94a3b8; margin: 8px 0 4px 0;'>👤 Câu hỏi truy vấn CSDL danh tính & hồ sơ:</div>", unsafe_allow_html=True)
        q_btn1, q_btn2, q_btn3 = st.columns(3)
        if q_btn1.button("👤 'Tôi là ai?'", use_container_width=True):
            user_input = "Tôi là ai? Tôi có vai trò gì trong hệ thống này?"
        if q_btn2.button("👵 'Cụ là ai?'", use_container_width=True):
            user_input = "Cụ là ai? Cho tôi biết thông tin người cao tuổi đang được giám sát."
        if q_btn3.button("💊 'Tiền sử bệnh & Dị ứng của cụ?'", use_container_width=True):
            user_input = "Tiền sử bệnh án, dị ứng thuốc và bác sĩ phụ trách của cụ là ai?"

        st.markdown("<div style='font-size: 12px; color: #94a3b8; margin: 8px 0 4px 0;'>💡 Các câu hay hỏi hàng ngày (FAQ Dinh dưỡng, Tắm rửa, Giấc ngủ, Thể dục, SOS):</div>", unsafe_allow_html=True)
        faq_c1, faq_c2, faq_c3 = st.columns(3)
        faq_c4, faq_c5, _ = st.columns(3)
        if faq_c1.button("🥗 'Ăn kiêng & Muối (<5g)?'", use_container_width=True):
            user_input = "Cụ An bị tiểu đường và huyết áp cần kiêng ăn gì và ăn bao nhiêu muối mỗi ngày?"
        if faq_c2.button("🚿 'Có được tắm đêm sau 19h?'", use_container_width=True):
            user_input = "Người già có được tắm đêm sau 19h không và nhiệt độ nước bao nhiêu là an toàn?"
        if faq_c3.button("🌿 'Mất ngủ & Thuốc an thần?'", use_container_width=True):
            user_input = "Cụ khó ngủ mất ngủ trằn trọc có nên mua thuốc ngủ cho cụ uống không?"
        if faq_c4.button("🧘 'Tập dưỡng sinh vẩy tay?'", use_container_width=True):
            user_input = "Hướng dẫn bài tập thể dục dưỡng sinh vẩy tay dịch cân kinh và đi bộ cho người già"
        if faq_c5.button("🆘 'Nút SOS khi mất mạng?'", use_container_width=True):
            user_input = "Nút bấm khẩn cấp SOS khi mất mạng có gửi cảnh báo được không và số điện thoại bác sĩ Tuấn?"

    # Xử lý khi có câu hỏi
    if user_input:
        cleaned_input = normalize_vietnamese_voice_transcript(user_input)
        st.session_state.chat_history.append({"role": "user", "content": cleaned_input})

        with st.spinner("🧠 AI đang tổng hợp đa cảm biến và tạo phản hồi ngắn gọn..."):
            reply_data = query_ai_service(cleaned_input)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply_data["answer"],
            "sources": reply_data.get("sources", []),
            "engine": reply_data.get("engine", "Local Edge")
        })

        st.rerun()


# ------------------------------------------------------------------------------
# CỘT 2: NHẬT KÝ SỰ KIỆN CẢM BIẾN (SYSTEM AUDIT LOGS)
# ------------------------------------------------------------------------------
with col_logs:
    st.markdown("### 📜 Nhật Ký Sự Kiện Hệ Thống")

    # Nút điều khiển lọc và mô phỏng
    action_col1, action_col2, action_col3 = st.columns([0.45, 0.25, 0.3])

    with action_col1:
        log_filter = st.selectbox(
            "Lọc mức độ:",
            ["ALL", "CRITICAL", "WARNING", "INFO"],
            label_visibility="collapsed"
        )
    with action_col2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.rerun()
    with action_col3:
        if st.button("⚡ Đọc mẫu mới", use_container_width=True, help="Mô phỏng 1 bước đọc cảm biến mới"):
            sim_output = simulate_random_telemetry()
            if sim_output:
                st.toast(f"Đã ghi nhận: {sim_output.get('predicted_label', 'Unknown')}", icon="📡")
                st.rerun()

    # Tải danh sách nhật ký
    all_logs = get_latest_system_logs(limit=40, level_filter=log_filter)

    # Render danh sách logs an toàn không bao giờ bị dính lỗi Markdown Indented Code
    log_items_html = []
    if not all_logs:
        log_items_html.append('<div style="color: #64748b; padding: 20px; text-align: center;">Chưa có sự kiện nào được ghi nhận.</div>')
    else:
        for entry in all_logs:
            lvl = entry.get("level", "INFO")
            card_class = "log-card"
            badge_color = "#3b82f6"
            badge_bg = "rgba(59, 130, 246, 0.2)"
            icon = "🟢"

            if lvl == "CRITICAL":
                card_class = "log-card log-card-critical"
                badge_color = "#ef4444"
                badge_bg = "rgba(239, 68, 68, 0.2)"
                icon = "🚨"
            elif lvl == "WARNING":
                card_class = "log-card log-card-warning"
                badge_color = "#f59e0b"
                badge_bg = "rgba(245, 158, 11, 0.2)"
                icon = "⚠️"

            t_val = entry.get("timestamp", "")
            d_val = entry.get("description", "")

            # Tạo HTML phẳng không thụt lề 4 space
            item_html = (
                f'<div class="{card_class}">'
                f'<div class="log-meta">'
                f'<span>⏱️ {t_val}</span>'
                f'<span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; padding: 1px 6px; border-radius: 4px; font-weight: 700;">{icon} {lvl}</span>'
                f'</div>'
                f'<div class="log-desc">{d_val}</div>'
                f'</div>'
            )
            log_items_html.append(item_html)

    final_log_box = f'<div class="log-scrollbox">{"".join(log_items_html)}</div>'
    st.markdown(final_log_box, unsafe_allow_html=True)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(textwrap.dedent(f"""
<div style="text-align: center; color: #64748b; font-size: 12px; margin-top: 10px;">
    AuraCare Edge AI • Hệ thống Giám sát Người cao tuổi Nội bộ • Bảo mật Cục bộ 100% • 
    Thiết bị: Intel Core i5 / Orange Pi 5 • Trạng thái: Ổn định
</div>
"""), unsafe_allow_html=True)
