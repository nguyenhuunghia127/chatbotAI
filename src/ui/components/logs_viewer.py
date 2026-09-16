# -*- coding: utf-8 -*-
"""
COMPONENT: KHUNG XEM NHẬT KÝ SỰ KIỆN CẢM BIẾN (SYSTEM AUDIT LOGS)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare
"""

from typing import Callable, List, Dict, Any, Protocol


class LogFetcher(Protocol):
    def __call__(self, limit: int = 40, level_filter: str = "ALL") -> List[Dict[str, Any]]:
        ...


def render_logs_viewer(
    st_module,
    get_logs_fn: Callable[..., List[Dict[str, Any]]] | LogFetcher,
    simulate_telemetry_fn: Callable[[], Any]
):
    """Vẽ bộ lọc nhật ký và danh sách thẻ sự kiện thời gian thực."""
    st_module.markdown("### 📜 Nhật Ký Sự Kiện Hệ Thống")

    # Nút điều khiển lọc và mô phỏng
    action_col1, action_col2, action_col3 = st_module.columns([0.45, 0.25, 0.3])

    with action_col1:
        log_filter = st_module.selectbox(
            "Lọc mức độ:",
            ["ALL", "CRITICAL", "WARNING", "INFO"],
            label_visibility="collapsed"
        )
    with action_col2:
        if st_module.button("🔄 Làm mới", use_container_width=True):
            st_module.rerun()
    with action_col3:
        if st_module.button("⚡ Đọc mẫu mới", use_container_width=True, help="Mô phỏng 1 bước đọc cảm biến mới"):
            sim_output = simulate_telemetry_fn()
            if sim_output:
                st_module.toast(f"Đã ghi nhận: {sim_output.get('predicted_label', 'Unknown')}", icon="📡")
                st_module.rerun()

    # Tải danh sách nhật ký
    all_logs = get_logs_fn(limit=40, level_filter=log_filter)

    # Render danh sách logs
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
    st_module.markdown(final_log_box, unsafe_allow_html=True)
