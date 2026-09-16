# -*- coding: utf-8 -*-
"""
COMPONENT: HỘP CHATBOT VÀ GỌI GIỌNG NÓI ĐA KÊNH
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

import hashlib
from typing import Callable, Any
from src.voice.voice_service import normalize_vietnamese_voice_transcript


def render_chat_interface(
    st_module,
    query_ai_service_fn: Callable[[str], dict],
    voice_service: Any
):
    """Vẽ toàn bộ giao diện chat, micro STT, các nút câu hỏi mẫu và modal RLHF."""
    st_module.markdown("### 💬 Trợ Lý AI Chăm Sóc Y Tế")

    # Các nút câu hỏi nhanh dạng thẻ ngang
    st_module.caption("⚡ Nhấp để hỏi nhanh về tình hình người cao tuổi:")
    q_col1, q_col2 = st_module.columns(2)
    selected_quick_q = None

    with q_col1:
        if st_module.button("🚨 Kiểm tra té ngã hôm nay", use_container_width=True):
            selected_quick_q = "Sáng nay ông có bị ngã không? Chi tiết thế nào và tôi phải làm gì?"
        if st_module.button("🌡️ Tình hình sốt & thân nhiệt", use_container_width=True):
            selected_quick_q = "Tình hình thân nhiệt và sốt của bà hôm nay như thế nào?"

    with q_col2:
        if st_module.button("💓 Cảnh báo nhịp tim bất thường", use_container_width=True):
            selected_quick_q = "Hôm nay có cảnh báo bất thường về tim mạch hoặc nhịp tim nhanh không?"
        if st_module.button("📋 Báo cáo tổng thể sức khỏe", use_container_width=True):
            selected_quick_q = "Hiện tại tình trạng sức khỏe chung của người cao tuổi ra sao?"

    # Container hiển thị đoạn chat
    chat_box = st_module.container(height=400)
    with chat_box:
        for idx, msg in enumerate(st_module.session_state.chat_history):
            if msg["role"] == "user":
                with st_module.chat_message("user", avatar="🧑‍💼"):
                    st_module.write(msg["content"])
            else:
                with st_module.chat_message("assistant", avatar="🤖"):
                    st_module.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        tags = []
                        if msg.get("crag_status"):
                            tags.append(f"🛡️ CRAG: `{msg['crag_status']}`")
                        if msg.get("graph_entities"):
                            tags.append(f"🕸️ Graph: `{', '.join(msg['graph_entities'][:3])}`")
                        if msg.get("tools_used"):
                            tags.append(f"🛠️ Tools: `{', '.join(msg['tools_used'])}`")
                        extra_tags = f" | {' | '.join(tags)}" if tags else ""
                        st_module.caption(f"📚 Nguồn: `{', '.join(msg['sources'])}` | Engine: `{msg.get('engine', 'AuraCare')}`{extra_tags}")

                    if msg.get("disclaimer"):
                        st_module.markdown(f"<div style='font-size: 11px; color: #94a3b8; font-style: italic; border-left: 2px solid #38bdf8; padding-left: 8px; margin: 4px 0 8px 0;'>{msg['disclaimer']}</div>", unsafe_allow_html=True)

                    # Thanh phản hồi Huấn luyện Tăng cường (RLHF & Active Learning)
                    if idx > 0:
                        prev_user_q = st_module.session_state.chat_history[idx - 1]["content"] if st_module.session_state.chat_history[idx - 1]["role"] == "user" else "Câu hỏi"
                        fb_c1, fb_c2, fb_c3 = st_module.columns([1, 1, 6])
                        if fb_c1.button("👍", key=f"fb_up_{idx}", help="Hài lòng - Thưởng điểm (+1) giúp AI ghi nhớ mẫu trả lời này"):
                            from src.training.rlhf_service import get_rlhf_service
                            get_rlhf_service().record_feedback(prompt=prev_user_q, response=msg["content"], rating=1)
                            st_module.toast("Đã ghi nhận 👍 (+1 điểm thưởng). AI đã ghi nhớ mẫu chuẩn này!", icon="🧠")
                        if fb_c2.button("👎", key=f"fb_down_{idx}", help="Chưa chuẩn - Phạt điểm (-1) và dạy câu trả lời đúng cho AI"):
                            st_module.session_state[f"edit_corr_{idx}"] = not st_module.session_state.get(f"edit_corr_{idx}", False)

                        if st_module.session_state.get(f"edit_corr_{idx}"):
                            with st_module.container():
                                st_module.markdown("<div style='font-size: 11.5px; color: #fbbf24; margin-top: 4px;'>✏️ <b>Dạy cho AI:</b> Nhập câu trả lời chuẩn xác mà bạn mong muốn:</div>", unsafe_allow_html=True)
                                corr_text = st_module.text_area("Câu trả lời đúng:", key=f"corr_inp_{idx}", height=70, placeholder="Ví dụ: Lúc 16h cụ sinh hoạt bình thường...")
                                if st_module.button("💾 Xác nhận Huấn Luyện AI", key=f"btn_corr_{idx}", use_container_width=True):
                                    if corr_text.strip():
                                        from src.training.rlhf_service import get_rlhf_service
                                        get_rlhf_service().record_feedback(prompt=prev_user_q, response=msg["content"], rating=-1, corrected_response=corr_text)
                                        st_module.session_state[f"edit_corr_{idx}"] = False
                                        st_module.success("✅ Đã cập nhật vào tập tri thức tăng cường! Lần sau hỏi câu này AI sẽ trả lời chuẩn xác theo bạn.")
                                        st_module.rerun()

    # Nhận câu hỏi qua chat_input
    user_input = st_module.chat_input("Nhập câu hỏi của bạn (ví dụ: Lúc 16h có ngã không?)...")
    if selected_quick_q:
        user_input = selected_quick_q

    # Nhận diện giọng nói Tiếng Việt trực tiếp từ Microphone trình duyệt
    st_module.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    mic_audio = st_module.audio_input("🎙️ Nhấn biểu tượng Micro để nói câu hỏi bằng Tiếng Việt...", label_visibility="collapsed")
    if mic_audio is not None:
        raw_bytes = mic_audio.read()
        if raw_bytes and len(raw_bytes) > 0:
            audio_hash = hashlib.md5(raw_bytes).hexdigest()
            if st_module.session_state.get("last_processed_audio") != audio_hash:
                st_module.session_state["last_processed_audio"] = audio_hash
                with st_module.spinner("🎙️ Hệ thống AI đang nhận diện giọng nói Tiếng Việt chuẩn xác..."):
                    transcribed = voice_service.transcribe_audio_bytes(raw_bytes, language="vi")
                    if transcribed and transcribed.strip():
                        user_input = transcribed.strip()
                        st_module.toast(f"Đã nhận diện: \"{user_input}\"", icon="🎙️")
                    else:
                        st_module.warning("⚠️ Chưa nhận diện được giọng nói rõ ràng. Vui lòng nói to và rõ hơn.")

    # Nút câu hỏi mẫu giọng nói (dự phòng nhanh)
    with st_module.expander("⚡ Bấm thử nhanh các câu hỏi mẫu (Sức khỏe & Hồ sơ CSDL)", expanded=False):
        st_module.markdown("<div style='font-size: 12px; color: #94a3b8; margin-bottom: 4px;'>🩺 Câu hỏi theo dõi sức khỏe & cảm biến:</div>", unsafe_allow_html=True)
        v_btn1, v_btn2, v_btn3 = st_module.columns(3)
        if v_btn1.button("🎤 'Sáng nay ông có ngã không?'", use_container_width=True):
            user_input = "Sáng nay ông có bị ngã không? Chi tiết thế nào và tôi phải làm gì?"
        if v_btn2.button("🎤 'Thân nhiệt bà có sốt không?'", use_container_width=True):
            user_input = "Tình hình thân nhiệt và sốt của bà hôm nay như thế nào?"
        if v_btn3.button("🎤 'Nhịp tim lúc 19h có sao không?'", use_container_width=True):
            user_input = "Hôm nay có cảnh báo bất thường về tim mạch hoặc nhịp tim nhanh không?"

        st_module.markdown("<div style='font-size: 12px; color: #94a3b8; margin: 8px 0 4px 0;'>👤 Câu hỏi truy vấn CSDL danh tính & hồ sơ:</div>", unsafe_allow_html=True)
        q_btn1, q_btn2, q_btn3 = st_module.columns(3)
        if q_btn1.button("👤 'Tôi là ai?'", use_container_width=True):
            user_input = "Tôi là ai? Tôi có vai trò gì trong hệ thống này?"
        if q_btn2.button("👵 'Cụ là ai?'", use_container_width=True):
            user_input = "Cụ là ai? Cho tôi biết thông tin người cao tuổi đang được giám sát."
        if q_btn3.button("💊 'Tiền sử bệnh & Dị ứng của cụ?'", use_container_width=True):
            user_input = "Tiền sử bệnh án, dị ứng thuốc và bác sĩ phụ trách của cụ là ai?"

        st_module.markdown("<div style='font-size: 12px; color: #94a3b8; margin: 8px 0 4px 0;'>💡 Các câu hay hỏi hàng ngày (FAQ Dinh dưỡng, Tắm rửa, Giấc ngủ, Thể dục, SOS):</div>", unsafe_allow_html=True)
        faq_c1, faq_c2, faq_c3 = st_module.columns(3)
        faq_c4, faq_c5, _ = st_module.columns(3)
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
        st_module.session_state.chat_history.append({"role": "user", "content": cleaned_input})

        with st_module.spinner("🧠 AI đang tổng hợp đa cảm biến và tạo phản hồi ngắn gọn..."):
            reply_data = query_ai_service_fn(cleaned_input)

        st_module.session_state.chat_history.append({
            "role": "assistant",
            "content": reply_data["answer"],
            "sources": reply_data.get("sources", []),
            "engine": reply_data.get("engine", "Local Edge"),
            "tools_used": reply_data.get("tools_used", []),
            "disclaimer": reply_data.get("disclaimer"),
            "crag_status": reply_data.get("crag_status"),
            "graph_entities": reply_data.get("graph_entities", []),
            "expanded_queries": reply_data.get("expanded_queries", [])
        })

        st_module.rerun()
