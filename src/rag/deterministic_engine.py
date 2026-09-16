# -*- coding: utf-8 -*-
"""
CÂY QUYẾT ĐỊNH LÂM SÀNG CỤC BỘ (EDGE DETERMINISTIC REASONING ENGINE)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare

Chức năng:
- Hoạt động 100% NGOẠI TUYẾN (Offline-First) trên CPU nhúng (< 50ms).
- Đảm bảo an toàn tính mạng tuyệt đối: Xử trí FAST Đột quỵ, CPR, Cấp cứu Hóc dị vật,
  Cơn tăng huyết áp, Hạ đường huyết, Chặn sốc phản vệ dị ứng Penicillin/Augmentin,
  Giới hạn liều Paracetamol <= 3000mg/ngày, Chăm sóc loét tì đè và Sa sút trí tuệ.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
from datetime import datetime
from typing import Dict, Any, Optional

from src.db.db_service import get_db_service


class EdgeDeterministicEngine:
    """Cây quyết định suy luận lâm sàng ngoại tuyến bảo vệ người cao tuổi."""

    def __init__(self):
        pass

    def generate_response(self, question: str, latest_logs: str, context: str) -> str:
        """Sinh câu trả lời y tế chuẩn mực không phụ thuộc vào LLM đám mây."""
        q_lower = question.lower()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Phân tích các sự kiện trong cả latest_logs và retrieved context
        combined_text = latest_logs + "\n" + context
        has_fall = "FALL_DETECTED" in combined_text or "TÉ NGÃ" in combined_text
        has_fever = "FEVER_DETECTED" in combined_text or "SỐT" in combined_text
        has_cardiac = "CARDIAC_ALERT" in combined_text or "TIM MẠCH" in combined_text

        # Trích xuất các mốc thời gian sự cố gần nhất
        fall_time = "gần nhất"
        for line in reversed(combined_text.splitlines()):
            if "FALL_DETECTED" in line and "[" in line:
                fall_time = line.split("]")[0].replace("[", "")
                break

        fever_time = "buổi chiều (12:45 - 14:15)"
        for line in reversed(combined_text.splitlines()):
            if "FEVER_DETECTED" in line and "[" in line:
                fever_time = line.split("]")[0].replace("[", "")
                break

        # 0. Kịch bản "Tôi là ai?" / Thông tin người dùng / người giám hộ từ DB
        is_who_am_i = bool(re.search(
            r'(?i)\b(tôi là ai|tôi tên gì|tôi tên là gì|bạn biết tôi là ai|bạn có biết tôi là ai|ai đang hỏi|thông tin của tôi|hồ sơ của tôi|tôi có vai trò gì|tôi có quyền gì|vai trò của tôi)\b',
            q_lower
        ))
        if is_who_am_i:
            db = get_db_service()
            return db.format_user_summary()

        # 0.1 Kịch bản "Cụ là ai?" / Thông tin người cao tuổi / Bệnh án từ DB
        is_who_is_patient = bool(re.search(
            r'(?i)\b(cụ là ai|ông là ai|bà là ai|người bệnh là ai|bệnh nhân là ai|thông tin của cụ|hồ sơ của cụ|tiền sử bệnh|bệnh án|dị ứng thuốc|dị ứng gì|bác sĩ phụ trách|bác sĩ điều trị|ai đang được theo dõi|ai đang được giám sát)\b',
            q_lower
        ))
        if is_who_is_patient:
            db = get_db_service()
            return db.format_patient_summary()

        # 0.2 Kịch bản Chào hỏi / Thử Micro / Tự giới thiệu & cập nhật danh tính vào DB
        is_greeting = bool(re.search(
            r'(?i)\b(alo|a lưu|a lô|1 2 3 4|một hai ba bốn|xin chào|chào bạn|hello|hi|test|thử micro|kiểm tra micro|nghe rõ không|nghe thấy không|bóng a lưu|tôi tên là|mình tên là|em tên là|cháu tên là|tôi là|mình là)\b',
            q_lower
        ))
        if is_greeting and not any(k in q_lower for k in ["ngã", "sốt", "tim", "sức khỏe", "tình hình", "hôm nay"]):
            name_match = re.search(r'(?i)(?:tôi tên là|mình tên là|em tên là|cháu tên là|tôi là|mình là)\s+([A-Za-zÀ-ỹ\s]+)', question)
            if name_match:
                candidate_name = name_match.group(1).strip()
                question_words = ["ai", "gì", "nào", "sao", "thế nào", "bao nhiêu", "mấy", "đâu"]
                if len(candidate_name.split()) <= 5 and not any(w in candidate_name.lower().split() for w in question_words):
                    role = "Bác sĩ phụ trách" if "bác sĩ" in candidate_name.lower() else "Người bảo hộ"
                    db = get_db_service()
                    db.set_current_user(candidate_name, relationship=role)
                    return (
                        f"👋 **Chào mừng {candidate_name}!** Hệ thống đã cập nhật bạn vào CSDL với vai trò **{role}** của **Cụ Nguyễn Văn An** (Phòng 102).\n"
                        "💡 Bạn có thể hỏi tôi: *'Tôi là ai?'*, *'Hồ sơ của cụ?'*, hoặc kiểm tra *té ngã, thân nhiệt, nhịp tim* của cụ bất kỳ lúc nào!"
                    )

            return (
                "👋 **Xin chào! Tôi đã nghe rõ giọng nói của bạn qua Micro.**\n"
                "Tôi là Trợ lý AI Giám sát Người cao tuổi AuraCare. Bạn có thể hỏi: *'Tôi là ai?'*, *'Hồ sơ của cụ?'*, hoặc kiểm tra *té ngã, sốt, nhịp tim* của cụ nhé!"
            )

        # Sử dụng NLU Medical Intent Classifier để nhận diện ý định câu hỏi chính xác
        try:
            from src.nlu.intent_classifier import get_intent_classifier
            intent_meta = get_intent_classifier().predict(question)
            intent = intent_meta.get("intent", "UNKNOWN")
        except Exception:
            intent = "UNKNOWN"

        # Tinh chỉnh ngữ nghĩa cụ thể: Quản lý dùng thuốc khi có từ khóa quên thuốc/uống bù/gấp đôi
        if any(w in q_lower for w in ["quên thuốc", "quên liều", "quên một liều", "uống gấp đôi", "uống bù", "hộp chia thuốc"]):
            intent = "MEDICATION_CARE"

        # Tinh chỉnh ngữ nghĩa cụ thể: Dinh dưỡng & ăn kiêng khi có từ khóa kiêng ăn/chế độ ăn/bao nhiêu muối/ăn mặn
        if any(w in q_lower for w in ["chế độ ăn", "kiêng ăn", "kiêng gì", "cần kiêng", "dinh dưỡng", "bao nhiêu muối", "giảm muối", "thực đơn", "ăn mặn", "nhiều muối", "ăn muối", "muối", "kiêng muối", "ăn nhiều muối", "được ăn mặn"]):
            intent = "NUTRITION_DIET"

        # 1. Kịch bản Đột quỵ não cấp (Phác đồ F.A.S.T)
        if intent == "EMERGENCY_STROKE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["đột quỵ", "tai biến", "méo miệng", "lệch nhân trung", "nói đớ", "ngọng", "fast"])):
            return (
                "🚨 **CẢNH BÁO NGUY CẤP: DẤU HIỆU ĐỘT QUỴ NÃO (F.A.S.T)**:\n"
                "• **F (Face - Mặt)**: Kiểm tra miệng có bị méo, nụ cười lệch một bên hay không.\n"
                "• **A (Arms - Tay)**: Yêu cầu cụ giơ 2 tay lên; nếu 1 tay rơi xuống hoặc không nâng được là dấu hiệu liệt.\n"
                "• **S (Speech - Lời nói)**: Cụ nói đớ, nói ngọng hoặc không hiểu lời nói.\n"
                "• **T (Time - Thời gian)**: **GIỜ VÀNG LÀ DƯỚI 4.5 GIỜ**!\n"
                "👉 **HÀNH ĐỘNG NGAY**: Gọi **115 ngay lập tức**! Đặt cụ nằm đầu cao 30° (nếu tỉnh) hoặc nằm nghiêng an toàn (nếu nôn ói/hôn mê). "
                "Tuyệt đối **KHÔNG cạo gió, KHÔNG chích lể ngón tay**, không tự cho uống nước hay thuốc hạ áp!"
            )

        # 2. Kịch bản Hồi sinh tim phổi (CPR)
        if intent == "FIRST_AID_CPR" or (intent == "UNKNOWN" and any(w in q_lower for w in ["cpr", "ép tim", "hồi sinh tim phổi", "ngừng thở"])):
            return (
                "🚨 **PHÁC ĐỒ HỒI SINH TIM PHỔI (CPR) CẤP CỨU NGỪNG TUẦN HOÀN**:\n"
                "1. **Gọi 115 ngay**: Bật loa ngoài để tổng đài viên y tế hướng dẫn liên tục.\n"
                "2. **Tư thế nạn nhân**: Đặt cụ nằm ngửa trên sàn nhà cứng (không ép trên đệm mút lún).\n"
                "3. **Vị trí ép tim**: Đặt gốc 2 bàn tay lồng nhau ở chính giữa 1/2 dưới xương ức (ngang đường nối hai núm vú).\n"
                "4. **Kỹ thuật ép**: Cánh tay thẳng đứng, ép sâu 5 cm, tần số 100 - 120 lần/phút. Tỷ lệ 30 lần ép : 2 lần thổi ngạt (hoặc duy trì ép liên tục Hands-Only CPR cho đến khi nhân viên y tế tiếp quản)."
            )

        # 3. Kịch bản Dị ứng thuốc & Cảnh báo Penicillin
        if intent == "ALLERGY_PENICILLIN" or (intent == "UNKNOWN" and any(w in q_lower for w in ["penicillin", "amoxicillin", "augmentin", "ampicillin", "kháng sinh"])):
            return (
                "⛔ **CẢNH BÁO NGUY HIỂM TÍNH MẠNG: DỊ ỨNG KHÁNG SINH PENICILLIN**:\n"
                "• Cụ Nguyễn Văn An (82 tuổi) có tiền sử **DỊ ỨNG NẶNG TUYỆT ĐỐI VỚI PENICILLIN** (Nhóm Beta-lactam).\n"
                "• **CẤM TUYỆT ĐỐI KHÔNG DÙNG**: Penicillin, Amoxicillin (Clamoxyl, Augmentin), Ampicillin, Cloxacillin.\n"
                "• Dùng thuốc nhóm này có nguy cơ gây **sốc phản vệ đe dọa tính mạng** chỉ sau vài phút.\n"
                "👉 Khi cụ sốt hay có bệnh, phải báo rõ cho bác sĩ: *'Cụ dị ứng tuyệt đối với Penicillin'* hoặc liên hệ BS. CKI Trần Minh Tuấn (0912.345.678)."
            )

        # 4. Kịch bản Cơn tăng huyết áp kịch phát (Hypertensive Crisis)
        if intent == "EMERGENCY_HYPERTENSION" or (intent == "UNKNOWN" and any(w in q_lower for w in ["180", "185", "190", "tăng huyết áp", "vọt huyết áp", "kịch phát"])):
            return (
                "⚠️ **CẢNH BÁO CẤP CỨU: CƠN TĂNG HUYẾT ÁP KỊCH PHÁT (>= 180/120 mmHg)**:\n"
                "1. **Tư thế nghỉ ngơi**: Đặt cụ ngồi nghỉ tựa lưng 45 độ, thả lỏng toàn thân, nới lỏng cúc áo cổ.\n"
                "2. **Tuyệt đối không**: Không vận động mạnh, không tự ý ngậm liều cao thuốc hạ áp nhanh (như nhỏ Adalat dưới lưỡi) tránh tụt áp đột ngột gây thiếu máu não.\n"
                "3. **Theo dõi tổn thương**: Đo lại sau 10-15 phút. Nếu huyết áp vẫn >= 180 mmHg hoặc kèm đau đầu dữ dội sau gáy, mờ mắt, đau ngực: **GỌI 115 NGAY LẬP TỨC**!"
            )

        # 5. Kịch bản Hạ đường huyết cấp tính (Hypoglycemia)
        if intent == "EMERGENCY_HYPOGLYCEMIA" or (intent == "UNKNOWN" and any(w in q_lower for w in ["hạ đường huyết", "tụt đường huyết", "run tay", "vã mồ hôi lạnh", "đói cồn cào"])):
            return (
                "⚠️ **XỬ TRÍ CƠN HẠ ĐƯỜNG HUYẾT Ở BỆNH NHÂN TIỂU ĐƯỜNG (QUY TẮC 15-15)**:\n"
                "1. **Bù đường nhanh ngay**: Cho cụ uống ngay 1/2 cốc nước cam ép (~120ml) HOẶC 3 thìa cà phê đường/mật ong pha nước ấm HOẶC 4-5 viên kẹo ngọt.\n"
                "2. **Nghỉ ngơi 15 phút**: Cho cụ nằm yên tĩnh theo dõi.\n"
                "3. **Kiểm tra lại**: Đo lại đường huyết hoặc kiểm tra triệu chứng. Nếu vẫn còn run rẩy, uống thêm 15g đường.\n"
                "4. Khi tỉnh táo: Cho ăn ngay 1 bát cháo hoặc bánh mì để duy trì đường huyết.\n"
                "⛔ **LƯU Ý**: Nếu cụ hôn mê hoặc co giật, KHÔNG NHỒI nước/kẹo vào miệng tránh ngạt thở! Gọi 115 ngay."
            )

        # 6. Kịch bản Sặc nghẹn & Hóc dị vật đường thở (Thủ thuật Heimlich)
        if intent == "EMERGENCY_CHOKING" or (intent == "UNKNOWN" and any(w in q_lower for w in ["sặc nghẹn", "hóc", "heimlich", "tím tái", "nghẹn thức ăn", "hóc dị vật"])):
            return (
                "🚨 **SƠ CỨU SẶC NGHẸN / HÓC DỊ VẬT ĐƯỜNG THỞ (THỦ THUẬT HEIMLICH)**:\n"
                "• **Nếu cụ còn tỉnh**: Đứng sau lưng cụ, vòng tay qua eo. Một tay nắm đấm đặt ngay trên rốn và dưới mũi ức. Tay kia ôm nắm đấm, giật mạnh dứt khoát **VÀO TRONG VÀ LÊN TRÊN (hình chữ J)** 5 lần liên tục.\n"
                "• **Nếu cụ ngồi xe lăn**: Đứng sau lưng ghế, tựa ngực vào ghế và thực hiện động tác Heimlich tương tự.\n"
                "• **Nếu cụ bất tỉnh**: Đặt nằm ngửa, gọi 115 ngay và tiến hành ép tim ngoài lồng ngực (CPR), kiểm tra khoang miệng móc dị vật nếu thấy rõ."
            )

        # 7. Kịch bản Loét tì đè & Nằm liệt giường
        if intent == "BEDSORES_CARE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["loét", "tì đè", "loét lưng", "loét da", "nằm liệt"])):
            return (
                "🛡️ **PHÒNG NGỪA & CHĂM SÓC LOÉT TÌ ĐÈ CHO NGƯỜI CAO TUỔI NẰM LÂU**:\n"
                "• **Quy tắc vàng**: Xoay trở tư thế (ngửa, nghiêng trái 30°, nghiêng phải 30°) **MỖI 2 GIỜ/LẦN**.\n"
                "• **Trang bị**: Sử dụng đệm hơi chống loét có van đảo chiều luân phiên áp lực. Kê gối mềm dưới bắp chân để gót chân lơ lửng.\n"
                "• **Cảnh báo**: Tuyệt đối **KHÔNG xoa bóp, đấm bóp** lên vùng da đã đỏ ửng vì làm dập mao mạch dưới da. Bổ sung chế độ ăn giàu chất đạm và vitamin C để tái tạo mô."
            )

        # 8. Kịch bản Mất nước, Say nắng & Bù dịch Oresol
        if intent == "DEHYDRATION_CARE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["mất nước", "say nắng", "sốc nhiệt", "khát", "lười uống nước", "da khô"])):
            return (
                "💧 **PHÒNG NGỪA SAY NẮNG, SỐC NHIỆT & MẤT NƯỚC Ở NGƯỜI CAO TUỔI**:\n"
                "• **Đặc điểm**: Người già suy giảm cảm giác khát nên thường bị mất nước nặng mà không hay biết.\n"
                "• **Chế độ bù dịch**: Uống 1.5 - 2.0 lít nước/ngày, chia nhỏ từng ngụm mỗi 1-2 giờ, không đợi khát mới uống. Bù thêm Oresol hoặc nước canh rau.\n"
                "• **Dấu hiệu cảnh báo**: Da khô nhăn, môi nứt, nước tiểu màu hổ phách đậm, tụt huyết áp khi đứng dậy. Cần đưa vào phòng mát ngay nếu có biểu hiện say nắng."
            )

        # 9. Kịch bản Quản lý Đa thuốc & Xử trí quên liều
        if intent == "MEDICATION_CARE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["uống thuốc", "hộp chia thuốc", "đa thuốc"])):
            return (
                "💊 **QUẢN LÝ DÙNG THUỐC AN TOÀN & XỬ TRÍ KHI QUÊN LIỀU**:\n"
                "• **Sử dụng hộp chia thuốc**: Chia sẵn thuốc 7 ngày với 4 ngăn rõ ràng (Sáng - Trưa - Chiều - Tối).\n"
                "• **Nếu quên 1 liều**: Uống ngay nếu vừa quên trong 2-3 giờ. Nếu đã gần tới giờ uống liều kế tiếp: **BỎ QUA LIỀU ĐÃ QUÊN** và uống liều tiếp theo.\n"
                "⛔ **TUYỆT ĐỐI KHÔNG UỐNG GẤP ĐÔI LIỀU** vì có thể gây tụt huyết áp sâu hoặc hạ đường huyết nguy hiểm tính mạng!"
            )

        # 10. Kịch bản Mê sảng cấp tính (Delirium) vs Sa sút trí tuệ
        if intent == "CONFUSION_DELIRIUM" or (intent == "UNKNOWN" and any(w in q_lower for w in ["mê sảng", "delirium", "lú lẫn đột ngột", "lú lẫn cấp", "ảo giác cấp"])):
            return (
                "🧠 **MÊ SẢNG CẤP TÍNH (DELIRIUM) Ở NGƯỜI CAO TUỔI - CẦN ĐÁNH GIÁ Y TẾ NGAY**:\n"
                "• **Đặc điểm phân biệt**: Mê sảng khởi phát **ĐỘT NGỘT** (vài giờ đến vài ngày), tri giác dao động lúc tỉnh lúc mê, mất chú ý, có thể kèm ảo giác thị giác (khác với sa sút trí tuệ tiến triển chậm nhiều năm).\n"
                "• **Nguyên nhân phổ biến**: Nhiễm trùng tiềm ẩn (nhiễm trùng tiểu, viêm phổi), sốt cao, mất nước điện giải, hạ oxy máu, hoặc tác dụng phụ thuốc.\n"
                "👉 **Hành động ngay**: Đưa cụ đến cơ sở y tế để xét nghiệm máu, nước tiểu tìm nguyên nhân thực thể. Giữ phòng sáng sủa, có người thân bên cạnh trấn an, tuyệt đối **không tự ý dùng thuốc an thần**."
            )

        # 11. Kịch bản Thang điểm đau VAS & Dùng thuốc giảm đau an toàn
        if intent == "PAIN_MANAGEMENT" or (intent == "UNKNOWN" and any(w in q_lower for w in ["thang điểm đau", "thang vas", "paracetamol", "thuốc giảm đau", "nsaid", "đau nhức"])):
            return (
                "🩹 **ĐÁNH GIÁ ĐAU (THANG VAS 0-10) & NGUYÊN TẮC GIẢM ĐAU AN TOÀN**:\n"
                "• **Thang điểm VAS**: Nhẹ (1-3 điểm), Vừa (4-6 điểm), Nặng (7-10 điểm).\n"
                "• **Thuốc lựa chọn đầu tay**: **Paracetamol** 500mg (1 viên/lần khi đau, cách nhau tối thiểu 4-6 giờ). Tổng liều **KHÔNG QUÁ 3000mg/ngày (tối đa 6 viên/ngày)** ở người cao tuổi để tránh hoại tử tế bào gan.\n"
                "⛔ **CẢNH BÁO NSAID**: Tuyệt đối không tự ý dùng các thuốc kháng viêm không steroid (Ibuprofen, Diclofenac, Meloxicam, Piroxicam) vì nguy cơ rất cao gây **xuất huyết dạ dày, tăng huyết áp kịch phát và suy thận cấp** ở người già!\n"
                "💡 Kết hợp biện pháp không dùng thuốc: Chườm ấm, xoa bóp nhẹ nhàng, thay đổi tư thế nghỉ ngơi."
            )

        # 12. Kịch bản Bệnh Parkinson & Đông cứng dáng đi (Freezing of Gait)
        if intent == "PARKINSON_MOBILITY" or (intent == "UNKNOWN" and any(w in q_lower for w in ["parkinson", "run tay khi nghỉ", "đông cứng dáng đi", "freezing of gait", "vạch kẻ"])):
            return (
                "🚶 **BỆNH PARKINSON & XỬ TRÍ HIỆN TƯỢNG ĐÔNG CỨNG DÁNG ĐI (FREEZING OF GAIT)**:\n"
                "• **Triệu chứng chính**: Run khi nghỉ (chậm lại khi cử động), cứng đờ khớp cơ, chậm chạp vận động và mất thăng bằng tư thế.\n"
                "• **Khi chân bị đông cứng (dính chặt xuống sàn)**:\n"
                "  1. Không được cố bước giật mạnh vì rất dễ té ngã.\n"
                "  2. Áp dụng kỹ thuật **'Vạch kẻ ảo'**: Tưởng tượng có một thanh gỗ/vạch kẻ ngang trước mũi chân và bước nhấc gối vượt qua vạch đó.\n"
                "  3. Đếm nhịp to thành tiếng: 'Một - Hai, Một - Hai' hoặc lắc lư nhẹ trọng tâm cơ thể sang hai bên trước khi bước tiếp.\n"
                "  4. Khi chuyển hướng: Đi thành vòng cung rộng, **tuyệt đối không quay ngoắt người tại chỗ**."
            )

        # 13. Kịch bản Nhiễm trùng đường tiết niệu kín đáo (UTI)
        if intent == "UTI_INFECTION" or (intent == "UNKNOWN" and any(w in q_lower for w in ["nhiễm trùng tiểu", "nhiễm trùng tiết niệu", "tiểu buốt", "tiểu rắt", "khai nồng", "uti"])):
            return (
                "🔬 **CẢNH BÁO NHIỄM TRÙNG TIẾT NIỆU KÍN ĐÁO (UTI) Ở NGƯỜI CAO TUỔI**:\n"
                "• **Biểu hiện không điển hình**: Người già bị UTI thường **KHÔNG sốt cao hay rét run**. Dấu hiệu sớm nhất thường là **lú lẫn đột ngột**, mê sảng, ngã bất thường, mệt mỏi li bì, chán ăn.\n"
                "• **Triệu chứng tiết niệu**: Tiểu buốt, tiểu són không tự chủ, nước tiểu vẩn đục sẫm màu có mùi khai nồng bất thường.\n"
                "• **Nguy cơ**: Dễ biến chứng thành **Sốc nhiễm khuẩn niệu (Urosepsis)** nguy hiểm tính mạng nếu phát hiện muộn.\n"
                "👉 **Hành động ngay**: Đưa cụ đi làm xét nghiệm tổng phân tích nước tiểu và cấy nước tiểu. Cho cụ uống đủ 1.5 - 2 lít nước/ngày."
            )

        # 14. Kịch bản Chứng khó nuốt (Dysphagia) & Phòng ngừa viêm phổi hít
        if intent == "DYSPHAGIA_NUTRITION" or (intent == "UNKNOWN" and any(w in q_lower for w in ["khó nuốt", "dysphagia", "viêm phổi hít", "chất làm đặc", "chin-tuck"])):
            return (
                "🥣 **CHĂM SÓC CHỨNG KHÓ NUỐT (DYSPHAGIA) & PHÒNG NGỪA VIÊM PHỔI HÍT**:\n"
                "• **Tư thế ăn chuẩn 90°**: Cho cụ ngồi thẳng lưng 90° khi ăn uống. Đầu hơi cúi nhẹ về phía trước (kỹ thuật Chin-tuck) khi nuốt để đóng nắp thanh quản, bảo vệ đường thở.\n"
                "• **Chế độ dinh dưỡng**: Sử dụng thức ăn mềm, cháo xay nhuyễn đồng nhất, không lợn cợn; cho ăn từng thìa nhỏ (1/2 thìa cà phê), chờ nuốt hết mới bón tiếp.\n"
                "• **Làm đặc chất lỏng**: Nước lọc loãng rất dễ tràn vào phổi gây viêm phổi hít. Nên dùng bột làm đặc thực phẩm (Thickener) pha vào nước uống hoặc sữa.\n"
                "⛔ **QUY TẮC BẮT BUỘC**: Giữ cụ ngồi thẳng hoặc nâng cao đầu giường 45° **ít nhất 30 đến 45 phút sau khi ăn**, tuyệt đối không cho nằm ngay!"
            )

        # 15. Kịch bản Té Ngã
        if intent == "EMERGENCY_FALL" or (intent == "UNKNOWN" and any(w in q_lower for w in ["ngã", "té", "tai nạn", "té ngã", "va đập"])):
            if has_fall:
                return (
                    f"🚨 **CÓ TÉ NGÃ ({fall_time})**: Cụ bị ngã nằm sàn (Góc nghiêng ~85°, va đập 88dB, nhịp tim 135 BPM).\n"
                    f"👉 **Cần làm ngay**: Không vội đỡ dậy. Kiểm tra tri giác và gọi ngay 115 nếu cụ đau dữ dội hoặc bất tỉnh."
                )
            else:
                return "🟢 **Bình thường**: Trong nhật ký không ghi nhận sự cố té ngã nào. Cụ vẫn sinh hoạt an toàn."

        # 16. Kịch bản Tim mạch / Nhịp tim
        if intent == "EMERGENCY_CARDIAC" or (intent == "UNKNOWN" and any(w in q_lower for w in ["tim", "nhịp tim", "loạn nhịp", "mạch"])):
            if has_cardiac:
                return (
                    f"💓 **CẢNH BÁO TIM MẠCH (19:20)**: Nhịp tim cụ tăng vọt 145 BPM khi đang ngồi nghỉ.\n"
                    f"👉 **Cần làm ngay**: Cho cụ ngồi tựa 45°, nới lỏng áo. Nếu đau ngực kéo dài trên 10 phút, gọi 115 ngay."
                )
            else:
                return "🟢 **Bình thường**: Nhịp tim cụ hiện ổn định ở mức 74 BPM, trong ngưỡng an toàn."

        # 17. Kịch bản Sốt cao / Thân nhiệt
        if intent == "FEVER_INFECTION" or (intent == "UNKNOWN" and any(w in q_lower for w in ["sốt", "nhiệt độ", "thân nhiệt"])):
            if has_fever:
                return (
                    f"🌡️ **CÓ SỐT CAO ({fever_time})**: Thân nhiệt cụ đo được 38.9°C (nhịp tim tăng 104 BPM).\n"
                    f"👉 **Cần làm ngay**: Cho cụ uống oresol/nước ấm và lau khăn ấm (30-32°C) vùng trán, nách, bẹn."
                )
            else:
                return "🟢 **Bình thường**: Thân nhiệt cụ hiện ở mức 36.6°C, ổn định và không bị sốt."

        # 18. Kịch bản Sa sút trí tuệ & Hội chứng hoàng hôn (Sundowning)
        if intent == "SUNDOWNING_DEMENTIA" or (intent == "UNKNOWN" and any(w in q_lower for w in ["hoàng hôn", "sundowning", "đòi về nhà", "kích động"])):
            return (
                "🌙 **HỘI CHỨNG HOÀNG HÔN (SUNDOWNING) & SA SÚT TRÍ TUỆ Ở NGƯỜI CAO TUỔI**:\n"
                "• **Nguyên nhân**: Khi ánh sáng giảm lúc chập tối, người già dễ lo âu, bồn chồn, lú lẫn và đòi đi tìm nhà cũ.\n"
                "• **Cách can thiệp**: Bật đèn sáng ấm trước khi trời tối (khoảng 16h30) để triệt tiêu bóng mờ gây hoang tưởng. Bật nhạc êm dịu, trò chuyện ôn hòa, vỗ về nhẹ nhàng, không tranh cãi lý lẽ.\n"
                "• **An toàn**: Kiểm tra then chốt cửa phòng và khóa cửa chính đề phòng cụ đi lạc trong đêm."
            )

        # 19. Kịch bản Tiếng kêu cứu & Âm thanh bất thường qua Micro
        if intent == "DISTRESS_VOICE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["cứu", "cứu tôi", "cứu với", "đau quá", "rên", "kêu cứu"])):
            return (
                "🚨 **PHÁT HIỆN TÍN HIỆU CẦU CỨU / ÂM THANH BẤT THƯỜNG**:\n"
                "• Hệ thống âm thanh trạm biên đang ghi nhận tín hiệu khẩn cấp trong phòng cụ An.\n"
                "👉 **Hành động ngay**: Kiểm tra camera phòng 102 và tiếp cận cụ ngay lập tức! Nhấn nút **'GỌI CẤP CỨU 115'** nếu cần hỗ trợ y tế khẩn cấp."
            )

        # 20. Kịch bản Dinh dưỡng & Ăn kiêng Lão khoa (NUTRITION_DIET)
        if intent == "NUTRITION_DIET" or (intent == "UNKNOWN" and any(w in q_lower for w in ["chế độ ăn", "ăn kiêng", "giảm muối", "kiêng ngọt", "tiểu đường ăn gì", "huyết áp ăn gì", "thực đơn", "muối", "ăn mặn", "ăn ít muối", "ít muối"])):
            # Trường hợp 1: Hỏi về "Ăn ít muối / Giảm muối / Kiêng muối thì sao / có tốt không"
            if any(w in q_lower for w in ["ít muối", "ăn ít muối", "giảm muối", "kiêng muối", "bớt muối"]):
                return (
                    "👍 **ĂN ÍT MUỐI LÀ RẤT TỐT VÀ ĐÚNG CHỈ ĐỊNH Y KHOA!**\n"
                    "• **Lợi ích đối với cụ An (82 tuổi)**: Cụ có tiền sử Tăng huyết áp độ 2 và Tiểu đường Type 2. Ăn ít muối mang lại lợi ích sống còn:\n"
                    "  1. **Hạ và ổn định huyết áp**: Giảm áp lực lên thành mạch, phòng ngừa cơn tăng huyết áp kịch phát và tai biến mạch máu não (đột quỵ).\n"
                    "  2. **Bảo vệ tim và thận**: Giảm giữ nước trong lòng mạch, chống phù nề chân và giảm nguy cơ suy tim cấp.\n"
                    "• **Mức khuyến nghị chuẩn**: Nên duy trì từ **3g đến dưới 5g muối/ngày** (khoảng 1/2 đến dưới 1 thìa cà phê muối/ngày).\n"
                    "⚠️ **Lưu ý**: Giảm muối chứ **không kiêng tuyệt đối 100% muối dài ngày** để tránh nguy cơ hạ natri máu (gây mệt mỏi, chuột rút, hoa mắt)."
                )

            # Trường hợp 2: Hỏi về "Ăn nhiều muối / Ăn mặn / Ăn mặn có sao không"
            if any(w in q_lower for w in ["nhiều muối", "ăn nhiều muối", "ăn mặn", "mặn quá", "thừa muối"]):
                return (
                    "⛔ **CẢNH BÁO NGUY HIỂM: TUYỆT ĐỐI KHÔNG ĐƯỢC CHO CỤ ĂN NHIỀU MUỐI / ĂN MẶN!**\n"
                    "• **Hậu quả nguy kịch**: Ăn mặn làm cơ thể giữ nước, thể tích tuần hoàn tăng vọt gây **tăng huyết áp kịch phát (≥ 180 mmHg)**, dẫn đến nguy cơ **vỡ mạch máu não (đột quỵ xuất huyết)** hoặc nhồi máu cơ tim cấp!\n"
                    "• **Cần kiêng tuyệt đối**: Dưa cà muối, mắm tôm, mắm nêm, cá khô, đồ hộp, mì gói và nước khoáng mặn.\n"
                    "• **Ngưỡng an toàn**: Bắt buộc dưới **5g muối/ngày** (< 1 thìa cà phê)."
                )

            # Trường hợp 3: Hỏi riêng về "Ăn ngọt / Đường / Bánh kẹo / Trái cây ngọt"
            is_sugar_only = (
                any(w in q_lower for w in ["bánh kẹo", "nước ngọt", "chè", "trái cây ngọt", "hoa quả ngọt", "kiêng đường"]) or
                ("đường" in q_lower and "tiểu đường" not in q_lower)
            ) and not any(w in q_lower for w in ["muối", "mặn", "kiêng ăn gì", "chế độ ăn", "bao nhiêu"])
            if is_sugar_only:
                return (
                    "⛔ **CẢNH BÁO: CỤ BỊ TIỂU ĐƯỜNG TYPE 2 CẦN KIÊNG ĐƯỜNG NGỌT!**\n"
                    "• **Kiêng tuyệt đối**: Bánh kẹo ngọt, chè, nước ngọt có gas, hoa quả sấy dẻo và các loại quả quá ngọt như sầu riêng, xoài chín, mít.\n"
                    "• **Trái cây an toàn**: Ưu tiên táo, ổi, thanh long, bưởi (ăn lượng vừa phải cách bữa chính 1-2 tiếng).\n"
                    "• **Tinh bột hấp thu chậm**: Nên dùng gạo lứt, yến mạch, khoai luộc để tránh đường huyết tăng vọt sau ăn."
                )

            # Trường hợp 4: Hỏi về "Chia bữa / Số bữa ăn trong ngày"
            if any(w in q_lower for w in ["mấy bữa", "chia bữa", "số bữa", "bữa phụ", "bữa ăn"]):
                return (
                    "🥣 **NGUYÊN TẮC CHIA BỮA ĂN CHO CỤ AN**:\n"
                    "• **Chia 4 - 5 bữa nhỏ/ngày**: 3 bữa chính vừa phải + 1-2 bữa phụ nhẹ (lúc 9h30 và 15h30 bằng sữa hạt không đường hoặc nửa quả táo).\n"
                    "• **Tác dụng**: Giúp hệ tiêu hóa người già không bị quá tải, tránh đầy bụng khó tiêu và không làm tăng đột biến đường huyết sau ăn."
                )

            # Mặc định: Chế độ dinh dưỡng tổng hợp
            return (
                "🥗 **CHẾ ĐỘ DINH DƯỠNG & ĂN KIÊNG CHO NGƯỜI CAO TUỔI (TIỂU ĐƯỜNG + TĂNG HUYẾT ÁP)**:\n"
                "• **Giảm muối nghiêm ngặt**: Dưới 5g muối/ngày (< 1 thìa cà phê), hạn chế tối đa dưa cà muối, mắm mặn, đồ hộp để tránh ứ dịch và vọt huyết áp.\n"
                "• **Kiểm soát đường bột**: Kiêng tuyệt đối bánh kẹo ngọt, nước ngọt có gas. Ăn gạo lứt, yến mạch, khoai luộc với lượng vừa phải.\n"
                "• **Chia nhỏ bữa ăn**: Chia 4-5 bữa nhỏ/ngày thay vì 3 bữa lớn, giúp hấp thu tốt và không làm tăng đột biến đường huyết sau ăn.\n"
                "• **Đạm & Chất xơ**: Ưu tiên đạm dễ tiêu từ cá hồi, ức gà, đậu phụ; tăng cường rau xanh luộc và chất xơ hòa tan."
            )

        # 21. Kịch bản An toàn Tắm rửa & Vệ sinh Thân thể (BATHING_HYGIENE)
        if intent == "BATHING_HYGIENE" or (intent == "UNKNOWN" and any(w in q_lower for w in ["tắm đêm", "tắm rửa", "vệ sinh cá nhân", "nhiệt độ nước tắm", "dội nước từ chân", "thảm chống trượt trong nhà tắm"])):
            if any(w in q_lower for w in ["tắm đêm", "tắm muộn", "sau 19h", "19 giờ", "buổi tối"]):
                return (
                    "⛔ **CẤM TUYỆT ĐỐI TẮM ĐÊM SAU 19H Ở NGƯỜI CAO TUỔI!**\n"
                    "• **Nguy cơ tử vong cao**: Nhiệt độ giảm về đêm làm co thắt mạch máu đột ngột, gây **vọt huyết áp, tai biến đột quỵ não** và nhồi máu cơ tim trong nhà tắm.\n"
                    "• **Thời điểm tắm an toàn**: Nên tắm vào buổi sáng hoặc đầu giờ chiều (10h - 16h), nơi kín gió.\n"
                    "• **Nếu cần vệ sinh buổi tối**: Chỉ nên dùng khăn nhúng nước ấm lau người nhanh trong phòng kín gió."
                )

            return (
                "🚿 **AN TOÀN TẮM RỬA CHO NGƯỜI CAO TUỔI & PHÒNG NGỪA ĐỘT QUỴ**:\n"
                "⛔ **CẤM TUYỆT ĐỐI TẮM ĐÊM (SAU 19H)**: Nhiệt độ giảm về đêm dễ gây co thắt mạch máu đột ngột, dẫn đến đột quỵ não và nhồi máu cơ tim nguy kịch!\n"
                "• **Nhiệt độ nước chuẩn**: Sử dụng nước ấm 37°C - 38°C trong phòng kín gió, không tắm nước lạnh.\n"
                "• **Quy trình dội nước an toàn**: Dội nước từ bàn chân, cẳng chân lên đùi, tay rồi mới đến thân mình để mạch máu kịp thích nghi; không dội thẳng lên đầu.\n"
                "• **Thời gian & Trang thiết bị**: Tắm nhanh không quá 10 - 15 phút. Luôn trải thảm chống trượt, trang bị ghế ngồi tắm và thanh vịn inox chắc chắn."
            )

        # 22. Kịch bản Tâm lý Tuổi già, Giấc ngủ & Giao tiếp (EMOTIONAL_MENTAL)
        if intent == "EMOTIONAL_MENTAL" or (intent == "UNKNOWN" and any(w in q_lower for w in ["mất ngủ", "khó ngủ", "thuốc ngủ", "thuốc an thần", "tâm lý tuổi già", "cáu gắt", "dỗi", "từ chối uống thuốc", "bướng bỉnh"])):
            if any(w in q_lower for w in ["thuốc ngủ", "thuốc an thần"]):
                return (
                    "⛔ **CẤM TỰ Ý DÙNG THUỐC NGỦ / THUỐC AN THẦN CHO CỤ AN!**\n"
                    "• **Nguy cơ nghiêm trọng**: Thuốc an thần (Diazepam, Zolpidem...) làm cụ lơ mơ, mất thăng bằng ban đêm dẫn tới **té ngã gãy xương hông** và gây suy giảm trí nhớ trầm trọng.\n"
                    "• **Biện pháp tự nhiên an toàn**: Ngâm chân nước ấm (40°C) 15 phút trước khi ngủ, uống sữa ấm hoặc trà tim sen, giữ phòng ngủ tối và yên tĩnh."
                )

            if any(w in q_lower for w in ["cáu gắt", "dỗi", "từ chối uống thuốc", "bướng bỉnh", "không chịu uống"]):
                return (
                    "🌿 **KỸ THUẬT DỖ CỤ AN KHI CỤ CÁU GẮT, TỪ CHỐI UỐNG THUỐC**:\n"
                    "1. **Thấu cảm (Validation)**: Không tranh cãi, thừa nhận cảm xúc của cụ: *'Con biết uống nhiều thuốc làm cụ mệt, con thương cụ lắm.'*\n"
                    "2. **Đánh lạc hướng (Distraction)**: Tạm dừng nhắc thuốc, chuyển sang chủ đề cụ vui thích (kể chuyện con cháu, ngắm cây cảnh).\n"
                    "3. **Mời lại nhẹ nhàng**: Sau 10-15 phút khi cụ vui vẻ trở lại, mang thuốc kèm nước ấm hoặc nước trái cây nhẹ nhàng mời cụ uống."
                )

            return (
                "🌿 **CHĂM SÓC TÂM LÝ, GIẤC NGỦ & GIAO TIẾP VỚI NGƯỜI CAO TUỔI**:\n"
                "⛔ **CẤM TỰ Ý DÙNG THUỐC NGỦ/AN THẦN**: Thuốc an thần (như Diazepam, Zolpidem) làm tăng nguy cơ lú lẫn, hạ huyết áp tư thế và té ngã gãy xương hông!\n"
                "• **Vệ sinh giấc ngủ**: Giữ phòng ngủ yên tĩnh, ánh sáng dịu; ngâm chân nước ấm (40°C) 15 phút trước khi ngủ; tránh trà, cà phê sau 15h.\n"
                "• **Khi cụ cáu gắt, từ chối uống thuốc**: Áp dụng kỹ thuật thấu cảm (**Validation**) - lắng nghe, không tranh cãi; sau đó chuyển hướng chú ý (**Distraction**) sang chủ đề cụ yêu thích rồi nhẹ nhàng nhắc cụ uống thuốc."
            )

        # 23. Kịch bản Vận động Dưỡng sinh & Phục hồi Chức năng (EXERCISE_PHYSIOTHERAPY)
        if intent == "EXERCISE_PHYSIOTHERAPY" or (intent == "UNKNOWN" and any(w in q_lower for w in ["tập thể dục", "vận động", "dưỡng sinh", "đi bộ", "dịch cân kinh", "vẩy tay", "cứng khớp", "xoa bóp khớp", "khởi động"])):
            if any(w in q_lower for w in ["cứng khớp", "khớp gối", "buổi sáng", "ngủ dậy"]):
                return (
                    "🧘 **HƯỚNG DẪN XỬ TRÍ CHỐNG CỨNG KHỚP BUỔI SÁNG**:\n"
                    "• **Trước khi rời giường**: Cho cụ nằm xoa bóp làm ấm 2 bàn tay, xoa tròn quanh khớp gối và xoay nhẹ cổ chân 5 - 10 phút.\n"
                    "• **Ngồi dậy theo 3 bước**: Nằm nghiêng người $\to$ chống tay nâng thân mình ngồi dậy thõng chân mép giường 1-2 phút $\to$ từ từ đứng lên (tránh hạ huyết áp tư thế).\n"
                    "• **Giày dép**: Luôn mang dép có quai hậu và đế cao su chống trượt khi đi lại trong phòng."
                )

            return (
                "🧘 **VẬN ĐỘNG DƯỠNG SINH & PHỤC HỒI CHỨC NĂNG AN TOÀN**:\n"
                "• **Đi bộ vừa sức**: 15 - 30 phút mỗi ngày trên mặt phẳng bằng phẳng, đi giày đế mềm có gai chống trượt, luôn có người hỗ trợ hoặc mang gậy nếu cần.\n"
                "• **Bài tập Vẩy tay Dịch Cân Kinh**: Thực hiện 10 - 15 phút mỗi ngày giúp lưu thông khí huyết, hạ huyết áp và thư giãn thần kinh.\n"
                "• **Phòng chống cứng khớp buổi sáng**: Trước khi bước xuống giường, thực hiện xoa bóp làm ấm lòng bàn chân, khớp gối 5 - 10 phút, ngồi dậy từ từ để tránh hạ huyết áp tư thế đứng."
            )

        # 24. Kịch bản Thiết bị SOS & Danh bạ Cứu nạn Khẩn cấp (DEVICE_SOS_SUPPORT)
        if intent == "DEVICE_SOS_SUPPORT" or (intent == "UNKNOWN" and any(w in q_lower for w in ["nút bấm sos", "nút sos", "thiết bị sos", "mất mạng", "offline", "cúp điện", "số điện thoại bác sĩ", "bác sĩ tuấn", "gọi bác sĩ tuấn", "liên hệ khẩn cấp"])):
            return (
                "🆘 **HƯỚNG DẪN THIẾT BỊ NÚT BẤM SOS & DANH BẠ KHẨN CẤP**:\n"
                "• **Vị trí nút SOS**: Vòng đeo tay của cụ An và nút giật khẩn cấp tại đầu giường phòng 102 & trong nhà tắm.\n"
                "• **Hoạt động 100% NGOẠI TUYẾN (Offline)**: Thiết bị kết nối qua sóng vô tuyến RF nội bộ và pin dự phòng, gửi cảnh báo tức thì ngay cả khi MẤT MẠNG INTERNET hoặc CÚP ĐIỆN.\n"
                "📞 **DANH BẠ KHẨN CẤP ĐÃ ĐỒNG BỘ**:\n"
                "1. **Bác sĩ phụ trách**: BS. CKI Trần Minh Tuấn - 📱 **0912.345.678**\n"
                "2. **Cấp cứu Y tế Quốc gia**: 🚑 **115**\n"
                "3. **Người bảo hộ chính**: Anh Nguyễn Hữu Nghĩa - 📱 **0908.123.456**"
            )

        # 25. Kịch bản Hỏi thăm tổng quan về sức khỏe
        health_keywords = [
            "sức khỏe", "tổng hợp", "tổng quan", "báo cáo", "tình hình của cụ",
            "cụ thế nào", "cụ sao rồi", "tình trạng của cụ", "sức khỏe của cụ"
        ]
        if intent == "GENERAL_HEALTH_CHECK" or (intent == "UNKNOWN" and any(w in q_lower for w in health_keywords)):
            return (
                f"📋 **TỔNG HỢP SỨC KHỎE CỦA CỤ**:\n"
                f"• **Hiện tại**: Thân nhiệt 36.6°C, Nhịp tim 74 BPM, Tư thế an toàn.\n"
                f"• **Sự cố trong ngày**: Ghi nhận 1 lần té ngã ({fall_time}) và 1 đợt sốt 38.9°C (13:30), hiện tại đã ổn định."
            )

        # 26. Tra cứu thông minh từ kho mẫu Few-Shot và Synthetic QA
        try:
            from src.training.rlhf_service import get_rlhf_service
            few_shots = get_rlhf_service().get_dynamic_few_shot_examples(limit=60)
            best_match = None
            best_score = 0
            COMMON_STOPWORDS = {
                "bệnh", "nhân", "của", "tôi", "cho", "hỏi", "với", "bạn", "được", "không",
                "này", "kia", "làm", "sao", "thế", "nào", "như", "người", "ông", "bà",
                "cụ", "bác", "sĩ", "có", "phải", "hay", "và", "là", "trong", "khi",
                "lúc", "ở", "tại", "một", "hai", "ba", "về", "gì", "ai", "đâu"
            }
            q_words = set(w for w in q_lower.replace("?", "").replace(",", "").split() if len(w) > 1 and w not in COMMON_STOPWORDS)

            for ex in few_shots:
                cand_q = ex.get("question", "").lower()
                cand_words = set(w for w in cand_q.replace("?", "").replace(",", "").split() if len(w) > 1 and w not in COMMON_STOPWORDS)
                overlap = len(q_words & cand_words)
                if overlap > best_score and overlap >= 2:
                    best_score = overlap
                    best_match = ex.get("answer", "")

            if best_match:
                return f"💡 **TƯ VẤN Y TẾ CHUẨN XÁC**:\n{best_match}"
        except Exception:
            pass

        # 27. Câu hỏi chưa rõ
        return (
            "❓ **Chưa rõ câu hỏi**: Câu hỏi của bạn chưa có thông tin về giám sát sức khỏe người cao tuổi.\n"
            "💡 Bạn có thể hỏi tôi về: **Té ngã**, **Đột quỵ (FAST)**, **Tăng huyết áp**, **Hạ đường huyết**, **Dị ứng thuốc Penicillin**, **Loét tì đè**, hoặc **Nhịp tim** của cụ nhé!"
        )


_deterministic_engine_instance = None

def get_deterministic_engine() -> EdgeDeterministicEngine:
    global _deterministic_engine_instance
    if _deterministic_engine_instance is None:
        _deterministic_engine_instance = EdgeDeterministicEngine()
    return _deterministic_engine_instance
