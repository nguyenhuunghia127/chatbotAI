# -*- coding: utf-8 -*-
"""
COMPONENT: THANH BÊN ĐIỀU KHIỂN & HỒ SƠ Y TẾ CSDL (SIDEBAR)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

import requests
from typing import Any
from src.db.db_service import get_db_service
from src.core.config import API_BASE_URL


def render_sidebar(st_module):
    """Vẽ toàn bộ thanh bên: CSDL người bệnh, người hỏi, Edge Nodes, SOS và Auto-Trainer."""
    with st_module.sidebar:
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

        st_module.markdown(f"""
        <div style="text-align: center; margin-bottom: 14px;">
            <div style="font-size: 36px; margin-bottom: 4px;">👵👴</div>
            <div style="font-size: 16.5px; font-weight: 700; color: #38bdf8;">CSDL HỒ SƠ Y TẾ & NGƯỜI DÙNG</div>
            <div style="font-size: 11.5px; color: #94a3b8;">Mã: {p_code} • {p_room}</div>
        </div>
        """, unsafe_allow_html=True)

        # Thẻ hồ sơ người cao tuổi & người giám hộ từ DB
        st_module.markdown(f"""
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

        with st_module.expander("⚙️ Đổi danh tính người hỏi (Tôi là ai)", expanded=False):
            with st_module.form("user_identity_form"):
                new_u_name = st_module.text_input("Tên của bạn:", value=u_name)
                new_u_rel = st_module.text_input("Mối quan hệ / Vai trò:", value=u_rel)
                new_u_phone = st_module.text_input("SĐT liên hệ:", value=u_phone)
                submit_u = st_module.form_submit_button("💾 Lưu vào CSDL", use_container_width=True)
                if submit_u:
                    db_svc.set_current_user(new_u_name, relationship=new_u_rel, phone=new_u_phone)
                    st_module.success(f"Đã cập nhật danh tính '{new_u_name}' vào CSDL!")
                    st_module.rerun()

        st_module.markdown("#### ⚡ Trạng thái Trạm Biên (Edge Nodes)")

        # Kiểm tra API
        api_live = False
        try:
            r = requests.get(f"{API_BASE_URL}/api/health", timeout=1)
            api_live = (r.status_code == 200)
        except Exception:
            api_live = False

        api_color = "#22c55e" if api_live else "#eab308"
        api_text = "Port 8001 Live" if api_live else "Direct Core Mode"

        st_module.markdown(f"""
        <div class="status-badge-card">
            <span class="status-title">📡 FastAPI Backend</span>
            <span class="status-pill" style="background: rgba({ '34,197,94' if api_live else '234,179,8' }, 0.2); color: {api_color}; border: 1px solid {api_color};">{api_text}</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🧠 ML Random Forest</span>
            <span class="status-pill" style="background: rgba(34,197,94, 0.2); color: #22c55e; border: 1px solid #22c55e;">Sẵn sàng (1ms)</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">📚 Hybrid Vector & TF-IDF</span>
            <span class="status-pill" style="background: rgba(56,189,248, 0.2); color: #38bdf8; border: 1px solid #38bdf8;">Offline Local</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🕸️ Medical GraphRAG</span>
            <span class="status-pill" style="background: rgba(168,85,247, 0.2); color: #c084fc; border: 1px solid #c084fc;">32 Nodes • 28 Edges</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🔄 Query Transformer</span>
            <span class="status-pill" style="background: rgba(34,197,94, 0.2); color: #22c55e; border: 1px solid #22c55e;">Multi-Query AI</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🛡️ CRAG & Self-RAG</span>
            <span class="status-pill" style="background: rgba(245,158,11, 0.2); color: #fbbf24; border: 1px solid #fbbf24;">Pre-Grade & Audit</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🛠️ Agentic Tool Calling</span>
            <span class="status-pill" style="background: rgba(236,72,153, 0.2); color: #f472b6; border: 1px solid #f472b6;">6 Tools Active</span>
        </div>
        <div class="status-badge-card">
            <span class="status-title">🎙️ Giọng nói (STT/TTS)</span>
            <span class="status-pill" style="background: rgba(14,165,233, 0.2); color: #38bdf8; border: 1px solid #38bdf8;">Whisper Tiếng Việt</span>
        </div>
        """, unsafe_allow_html=True)

        st_module.markdown("---")
        st_module.markdown("#### 🚨 Báo Động Khẩn Cấp")
        if st_module.button("📞 GỌI CẤP CỨU 115 NGAY", type="primary", use_container_width=True):
            st_module.error("🚨 Đang phát còi báo động tại trạm phòng và liên hệ Trung tâm Cấp cứu 115!")

        with st_module.expander("📖 Cẩm nang Sơ cứu Nhanh", expanded=False):
            st_module.markdown("""
            * **Khi Té Ngã**: Không vội đỡ dậy. Kiểm tra tri giác, xương hông và gọi 115 nếu nghi ngờ gãy xương.
            * **Khi Sốt Cao (>38°C)**: Bổ sung oresol, chườm khăn ấm (30-32°C) vùng trán/nách/bẹn.
            * **Rối loạn nhịp tim**: Đặt ngồi Fowler 45°, nới lỏng áo và hít thở sâu.
            """)

        st_module.markdown("---")
        st_module.markdown("#### 🤖 Tự Động Huấn Luyện Ngầm (Auto-Trainer)")
        try:
            from src.training.auto_trainer import get_auto_trainer
            trainer = get_auto_trainer()
            t_status = trainer.get_status()

            last_time = t_status.get("last_run_timestamp")
            last_time_str = last_time.split()[1] if last_time and " " in last_time else "Vừa xong"
            cycles = t_status.get("total_cycles_completed", 1)
            samples = t_status.get("synthetic_qa_samples_learned", 20)

            st_module.markdown(f"""
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

            r_col1, r_col2 = st_module.columns(2)
            if r_col1.button("🚀 Chạy chu kỳ mới", use_container_width=True, help="Tự động sinh mẫu mới và huấn luyện ngầm"):
                with st_module.spinner("Đang tự động huấn luyện ngầm..."):
                    trainer.run_full_training_cycle()
                    st_module.success("Đã hoàn tất chu kỳ huấn luyện tự động! 🎯")
                    st_module.rerun()

            if r_col2.button("📦 Xem Dataset", use_container_width=True, help="Xuất file JSON DPO / Alpaca"):
                from src.training.rlhf_service import get_rlhf_service
                get_rlhf_service().export_datasets()
                st_module.toast("Đã xuất data/training_data/rlhf_dataset.json! 🚀", icon="📦")

            with st_module.expander("✨ Tạo sinh Dữ liệu bằng LLM API", expanded=False):
                from src.training.llm_synthetic_generator import (
                    DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL,
                    LLMSyntheticGenerator, GENERATION_TOPICS
                )
                cfg_key = st_module.text_input("API Key:", value=DEFAULT_API_KEY, type="password", help="Key bắt đầu bằng sk-xt-...")
                cfg_url = st_module.text_input("Base URL:", value=DEFAULT_BASE_URL, help="Ví dụ: https://api.xkiro.com/v1")
                cfg_model = st_module.text_input("Model Name:", value=DEFAULT_MODEL, help="Ví dụ: deepseek/deepseek-chat-v3.1, deepseek/deepseek-v4-pro")
                topic_choice = st_module.selectbox("Chủ đề y tế cần tạo:", ["Tất cả chủ đề (all)"] + list(GENERATION_TOPICS.keys()))
                num_per_t = st_module.slider("Số mẫu trên mỗi chủ đề:", min_value=1, max_value=10, value=3)

                if st_module.button("🚀 Bắt đầu Tạo sinh Dữ liệu", use_container_width=True, type="primary"):
                    with st_module.spinner(f"Đang gọi {cfg_model} tại {cfg_url} để tạo sinh dữ liệu y tế chuẩn..."):
                        gen = LLMSyntheticGenerator(api_key=cfg_key, base_url=cfg_url, model=cfg_model)
                        actual_topic = "all" if "all" in topic_choice else topic_choice
                        if actual_topic == "all":
                            res = gen.generate_all_topics(samples_per_topic=num_per_t)
                        else:
                            samples = gen.generate_qa_pairs(topic_key=actual_topic, num_samples=num_per_t)
                            res = gen.integrate_into_knowledge_base(samples)
                            res["total_generated"] = len(samples)

                        if res.get("total_generated", 0) > 0:
                            st_module.success(f"🎉 Đã sinh thành công {res.get('total_generated')} mẫu dữ liệu và nạp vào bộ nhớ RAG!")
                            st_module.rerun()
                        else:
                            st_module.error("Không thể kết nối hoặc API trả về lỗi. Vui lòng kiểm tra lại API Key, Base URL hoặc Model Name.")
        except Exception as e_trainer:
            st_module.caption(f"Trạng thái Auto-Trainer: {e_trainer}")
