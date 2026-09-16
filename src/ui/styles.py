# -*- coding: utf-8 -*-
"""
MODULE GIAO DIỆN & CSS CAO CẤP (ULTRA-PREMIUM HEALTHCARE THEME)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

import textwrap

CUSTOM_CSS = textwrap.dedent("""
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


def apply_custom_styles(st_module):
    """Áp dụng toàn bộ CSS giao diện tùy biến vào trang Streamlit."""
    st_module.markdown(CUSTOM_CSS, unsafe_allow_html=True)
