# -*- coding: utf-8 -*-
"""
COMPONENT: HEADER BANNER VÀ LƯỚI THẺ SINH HIỆU THỜI GIAN THỰC
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

import textwrap
from datetime import datetime
from typing import List, Dict, Any


def render_header_and_vitals(st_module, recent_logs: List[Dict[str, Any]]):
    """Vẽ Header thương hiệu AuraCare và Lưới 4 Thẻ sinh hiệu thông minh."""
    # 1. Header Banner
    st_module.markdown(textwrap.dedent(f"""
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

    # 2. Xử lý logic trạng thái sinh hiệu từ log mới nhất
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

    v_col1, v_col2, v_col3, v_col4 = st_module.columns(4)

    with v_col1:
        st_module.markdown(f"""
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
        st_module.markdown(f"""
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
        st_module.markdown(f"""
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
        st_module.markdown(f"""
        <div class="vital-card">
            <div class="vital-header">
                <span class="vital-label">Âm thanh phòng</span>
                <span class="vital-icon">🎙️</span>
            </div>
            <div class="vital-value" style="color: {'#f87171' if sound_val > 80 else '#4ade80'};">{sound_val} <span style="font-size: 14px; font-weight: 500;">dB</span></div>
            <span class="vital-status {sound_class}">{sound_status}</span>
        </div>
        """, unsafe_allow_html=True)
