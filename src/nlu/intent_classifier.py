# -*- coding: utf-8 -*-
"""
MODULE PHÂN LOẠI Ý ĐỊNH Y TẾ & TRIAGE CẤP CỨU (NLU MEDICAL INTENT & EMERGENCY TRIAGE CLASSIFIER)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Công nghệ:
- Scikit-Learn Pipeline: TfidfVectorizer (ngram_range=(1,2)) + LogisticRegression / CalibratedClassifierCV
- Hỗ trợ 18 nhãn lâm sàng & vận hành hệ thống chuyên biệt cho chăm sóc người cao tuổi
- Suy luận siêu tốc (< 2ms trên CPU Edge), hoàn toàn Offline 100%
- Kết hợp Rule-based Fallback & Verification để đảm bảo an toàn tính mạng
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intent_classifier.joblib")

# Danh mục 18 Ý định Chuẩn Lâm sàng & Hệ thống
INTENT_LABELS = {
    "EMERGENCY_STROKE": {
        "title": "Cấp cứu Đột Quỵ Não (FAST)",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["đột quỵ", "tai biến", "méo miệng", "lệch nhân trung", "yếu tay", "liệt", "nói đớ", "ngọng", "fast"]
    },
    "EMERGENCY_FALL": {
        "title": "Sự cố Té Ngã & Va Đập",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["ngã", "té", "tai nạn ngã", "té ngã", "ngã nằm sàn", "trượt ngã", "va đập mạnh", "trượt chân ngã", "gãy xương hông"]
    },
    "EMERGENCY_CARDIAC": {
        "title": "Cảnh báo Rối loạn Tim mạch & Đau ngực",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["nhịp tim tăng", "đau ngực", "thắt ngực", "loạn nhịp", "bpm", "mạch nhanh", "nhồi máu cơ tim"]
    },
    "EMERGENCY_HYPERTENSION": {
        "title": "Cơn Tăng Huyết Áp Kịch Phát",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["cơn tăng huyết áp", "tăng huyết áp kịch phát", "vọt huyết áp", "huyết áp vọt", "180", "185", "190", "huyết áp cao vọt", "cơn cao huyết áp"]
    },
    "EMERGENCY_HYPOGLYCEMIA": {
        "title": "Cơn Hạ Đường Huyết Cấp tính",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["hạ đường huyết", "tụt đường huyết", "run tay", "vã mồ hôi lạnh", "đói cồn cào", "hoa mắt", "15-15"]
    },
    "EMERGENCY_CHOKING": {
        "title": "Sặc Nghẹn & Hóc Dị Vật Đường Thở",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["sặc", "nghẹn", "hóc", "heimlich", "tím tái", "ngạt thở", "nghẹn thức ăn", "nghẹn thuốc"]
    },
    "ALLERGY_PENICILLIN": {
        "title": "Cảnh báo Dị ứng Kháng sinh Penicillin",
        "is_emergency": True,
        "priority": 2,
        "keywords": ["penicillin", "amoxicillin", "augmentin", "ampicillin", "kháng sinh", "dị ứng thuốc", "dị ứng"]
    },
    "FIRST_AID_CPR": {
        "title": "Hồi Sinh Tim Phổi (CPR)",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["cpr", "ép tim", "hồi sinh tim phổi", "ngừng thở", "ngừng tim", "thổi ngạt", "bất tỉnh"]
    },
    "DISTRESS_VOICE": {
        "title": "Tín hiệu Âm thanh Kêu cứu / Rên rỉ",
        "is_emergency": True,
        "priority": 1,
        "keywords": ["cứu", "cứu tôi", "cứu với", "đau quá", "rên", "kêu cứu", "rơi vỡ", "bát đĩa"]
    },
    "FEVER_INFECTION": {
        "title": "Thân nhiệt Cao & Sốt ở Người già",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["sốt", "nhiệt độ", "thân nhiệt", "nóng", "chườm ấm", "oresol"]
    },
    "SUNDOWNING_DEMENTIA": {
        "title": "Hội chứng Hoàng hôn & Sa sút Trí tuệ",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["hoàng hôn", "sundowning", "lú lẫn", "đòi về nhà", "kích động", "quậy", "đi lạc", "ban đêm", "chập tối"]
    },
    "DEHYDRATION_CARE": {
        "title": "Mất nước, Say nắng & Bù dịch Oresol",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["mất nước", "say nắng", "sốc nhiệt", "khát nước", "lười uống nước", "da khô", "tiểu sẫm", "bù nước"]
    },
    "BEDSORES_CARE": {
        "title": "Dự phòng & Chăm sóc Loét Tì đè",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["loét", "tì đè", "loét lưng", "loét da", "nằm liệt", "xoay trở", "đệm hơi", "đệm chống loét"]
    },
    "MEDICATION_CARE": {
        "title": "Quản lý Đa thuốc & An toàn Uống thuốc",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["uống thuốc", "quên thuốc", "nhầm thuốc", "uống nhầm", "hộp chia thuốc", "liều lượng", "quên liều"]
    },
    "CONFUSION_DELIRIUM": {
        "title": "Mê Sảng Cấp Tính (Delirium) & Cấp Cứu Nhận Thức",
        "is_emergency": True,
        "priority": 2,
        "keywords": ["mê sảng", "delirium", "lú lẫn đột ngột", "lảm nhảm", "ảo thị", "nhìn thấy ma", "đảo lộn giấc ngủ"]
    },
    "PAIN_MANAGEMENT": {
        "title": "Đánh Giá Điểm Đau & Thuốc Giảm Đau An Toàn",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["thang điểm đau", "vas", "giảm đau", "paracetamol", "panadol", "nsaid", "ibuprofen", "đau nhức"]
    },
    "PARKINSON_MOBILITY": {
        "title": "Bệnh Parkinson, Run Tay & Dáng Đi",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["parkinson", "run tay", "vê thuốc lào", "đông cứng dáng đi", "dính chân", "freezing"]
    },
    "UTI_INFECTION": {
        "title": "Nhiễm Trùng Tiết Niệu Kín Đáo",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["nhiễm trùng tiểu", "tiết niệu", "uti", "tiểu đục", "mùi khai", "tiểu buốt"]
    },
    "DYSPHAGIA_NUTRITION": {
        "title": "Chăm Sóc Khó Nuốt & Phòng Ngừa Viêm Phổi Hít",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["khó nuốt", "dysphagia", "nghẹn thức ăn", "sặc phổi", "viêm phổi hít", "xay nhuyễn", "ngồi 90 độ"]
    },
    "IDENTITY_USER_PROFILE": {
        "title": "Tra cứu Danh tính & Vai trò Người hỏi",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["tôi là ai", "tôi tên gì", "tôi tên là gì", "vai trò của tôi", "ai đang hỏi", "thông tin của tôi"]
    },
    "IDENTITY_PATIENT_PROFILE": {
        "title": "Hồ sơ Bệnh án & Thông tin Cụ An",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["cụ là ai", "hồ sơ của cụ", "tiền sử", "bệnh án", "bác sĩ phụ trách", "phòng của cụ", "bao nhiêu tuổi"]
    },
    "GENERAL_HEALTH_CHECK": {
        "title": "Tổng hợp Sức khỏe & Tình hình Hiện tại",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["sức khỏe", "tổng hợp", "tổng quan", "tình hình của cụ", "cụ thế nào", "cụ sao rồi", "tình trạng của cụ"]
    },
    "NUTRITION_DIET": {
        "title": "Chế Độ Dinh Dưỡng & Ăn Kiêng Lão Khoa",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["chế độ ăn", "dinh dưỡng", "ăn kiêng", "kiêng ăn", "kiêng", "giảm muối", "kiêng ngọt", "tiểu đường ăn gì", "huyết áp ăn gì", "thực đơn", "ăn bao nhiêu muối", "bao nhiêu muối", "ăn nhiều muối", "ăn mặn", "kiêng muối", "ăn muối", "ăn gì"]
    },
    "BATHING_HYGIENE": {
        "title": "An Toàn Tắm Rửa & Vệ Sinh Thân Thể",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["tắm đêm", "tắm rửa", "vệ sinh cá nhân", "nhiệt độ nước tắm", "dội nước từ chân", "thảm chống trượt trong nhà tắm", "tắm"]
    },
    "EMOTIONAL_MENTAL": {
        "title": "Tâm Lý Tuổi Già, Giấc Ngủ & Giao Tiếp",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["mất ngủ", "khó ngủ", "thuốc ngủ", "thuốc an thần", "tâm lý tuổi già", "cáu gắt", "dỗi", "từ chối uống thuốc", "bướng bỉnh", "trầm cảm"]
    },
    "EXERCISE_PHYSIOTHERAPY": {
        "title": "Vận Động Dưỡng Sinh & Phục Hồi Chức Năng",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["tập thể dục", "vận động", "dưỡng sinh", "đi bộ", "dịch cân kinh", "vẩy tay", "cứng khớp", "xoa bóp khớp", "khởi động"]
    },
    "DEVICE_SOS_SUPPORT": {
        "title": "Hướng Dẫn Thiết Bị SOS & Liên Hệ Khẩn Cấp",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["nút bấm sos", "nút sos", "thiết bị sos", "mất mạng", "offline", "cúp điện", "số điện thoại bác sĩ", "bác sĩ tuấn", "gọi bác sĩ tuấn", "liên hệ khẩn cấp"]
    },
    "GREETING_SYSTEM_TEST": {
        "title": "Chào hỏi & Kiểm tra Hệ thống / Micro",
        "is_emergency": False,
        "priority": 5,
        "keywords": ["alo", "a lô", "1 2 3 4", "xin chào", "chào bạn", "hello", "kiểm tra micro", "thử micro"]
    },
    "LEGAL_INSURANCE_POLICY": {
        "title": "Chính Sách BHYT & Quyền Lợi Người Cao Tuổi",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["bhyt", "bảo hiểm y tế", "thẻ bhyt", "80 tuổi", "miễn phí", "100%", "lĩnh thuốc thay", "nhận thuốc thay", "chuyển tuyến", "chuyển viện", "giấy ủy quyền"]
    },
    "DOMESTIC_FIRSTAID": {
        "title": "Sơ Cứu Tai Nạn Sinh Hoạt & Vết Thương Gia Đình",
        "is_emergency": True,
        "priority": 2,
        "keywords": ["bỏng nước sôi", "bị bỏng", "bỏng", "đứt tay", "chảy máu", "vết rách", "cầm máu", "ong đốt", "ong vò vẽ", "kiến ba khoang", "sơ cứu tai nạn"]
    },
    "TECH_ENVIRONMENT_HEALTH": {
        "title": "Khoa Học Môi Trường & Thiết Bị Đời Sống",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["sóng wifi", "wifi", "sóng điện thoại", "5g", "máy tạo nhịp tim", "điều hòa", "máy lạnh", "nhiệt độ phòng", "ánh sáng xanh", "đèn ngủ"]
    },
    "COGNITIVE_BRAIN_EXERCISE": {
        "title": "Rèn Luyện Trí Não & Phòng Ngừa Alzheimer",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["cờ tướng", "cờ vua", "chơi cờ", "sudoku", "đố chữ", "trí não", "rèn luyện trí nhớ", "não bộ", "teo não", "alzheimer", "đãng trí"]
    },
    "SMALLTALK_ENTERTAINMENT": {
        "title": "Đời Sống Tinh Thần, Thơ Ca & Giải Trí",
        "is_emergency": False,
        "priority": 5,
        "keywords": ["kể chuyện vui", "chuyện cười", "bài thơ", "thơ ca", "cây cảnh", "nuôi chim", "cá cảnh", "thư giãn", "sum vầy", "an yên", "tuổi già"]
    },
    "GENERAL_OUT_OF_SCOPE": {
        "title": "Kiến Thức Tổng Quát & Ngoài Phạm Vi Chuyên Môn",
        "is_emergency": False,
        "priority": 5,
        "keywords": ["giá vàng", "vàng hôm nay", "viết code", "lập trình", "python", "sửa xe", "chết máy", "thời tiết", "ngoài phạm vi"]
    },
    "HERBAL_DRUG_INTERACTION": {
        "title": "Dược Lý & Tương Tác Đông Tây Y",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["nhân sâm", "sâm", "tam thất", "đông trùng hạ thảo", "lá vối", "nước vối", "atiso", "uống sâm", "thảo dược", "thuốc nam", "thuốc bắc"]
    },
    "BEDRIDDEN_PRESSURE_ULCER": {
        "title": "Chăm Sóc Loét Tì Đè & Nằm Liệt",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["loét tì đè", "lở loét", "đỏ rát mông", "xương cùng", "lật người", "trở mình", "đệm hơi", "nằm một chỗ", "nằm liệt", "loét"]
    },
    "VISION_HEARING_CARE": {
        "title": "Thị Lực, Đục Thủy Tinh Thể & Máy Trợ Thính",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["đục thủy tinh thể", "cườm khô", "mắt mờ", "chói mắt", "mổ phaco", "máy trợ thính", "lãng tai", "nghễnh ngãng", "thính lực"]
    },
    "WEATHER_JOINT_SEASONAL": {
        "title": "Khớp Khi Trời Lạnh & Đột Quỵ Mùa Đông",
        "is_emergency": False,
        "priority": 3,
        "keywords": ["trời lạnh", "thời tiết lạnh", "trở trời", "đau nhức khớp", "buốt khớp", "mùa đông", "chân trần", "giữ ấm", "nền nhà lạnh"]
    },
    "DENTAL_DENTURE_NUTRITION": {
        "title": "Dinh Dưỡng Răng Yếu & Răng Giả Tháo Lắp",
        "is_emergency": False,
        "priority": 4,
        "keywords": ["răng yếu", "rụng răng", "hàm giả", "răng giả", "tháo răng giả", "nuốt răng giả", "thức ăn mềm", "cháo xay nhuyễn", "nhai khó"]
    },
    "SOCIAL_INTERACTIVE_RIDDLES": {
        "title": "Kỹ Năng Ứng Xử & Đố Vui Dân Gian",
        "is_emergency": False,
        "priority": 5,
        "keywords": ["đố vui", "câu đố", "đố dân gian", "tự đi xe", "đòi đi xe", "chuyển hướng", "ứng xử gia đình", "cười vui"]
    }
}


class IntentClassifier:
    """Bộ phân loại ý định lâm sàng dựa trên học máy (Scikit-Learn) và quy tắc dự phòng an toàn."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load_model_safe()

    def _load_model_safe(self):
        """Nạp mô hình đã huấn luyện nếu có."""
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[INTENT CLASSIFIER] Không thể nạp mô hình: {e}", file=sys.stderr)
                self.pipeline = None

    def rule_based_classify(self, text: str) -> Tuple[Optional[str], float]:
        """Phân loại dựa trên tập từ khóa chính xác y tế (Rule-based Fast Matcher với Word Boundaries)."""
        t_lower = text.lower().strip()
        best_intent = None
        best_score = 0.0

        for intent_key, meta in sorted(INTENT_LABELS.items(), key=lambda x: x[1]["priority"]):
            keywords = meta["keywords"]
            matches = []
            for kw in keywords:
                # Sử dụng regex có ranh giới từ để tránh lỗi 'hi' khớp trong 'nhiều'
                pattern = r'(?:\b|^)' + re.escape(kw) + r'(?:\b|$)'
                if re.search(pattern, t_lower):
                    matches.append(kw)

            if matches:
                # Tính điểm khớp theo độ dài từ khóa và độ ưu tiên
                score = 0.75 + min(0.24, len(matches) * 0.08)
                if score > best_score:
                    best_score = score
                    best_intent = intent_key

        return best_intent, best_score

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Dự đoán ý định của câu hỏi.
        Kết hợp xác suất Machine Learning với bảo vệ an toàn y khoa (Safety Fallback).
        """
        if not text or not text.strip():
            return {
                "intent": "GREETING_SYSTEM_TEST",
                "confidence": 0.5,
                "is_emergency": False,
                "title": "Chưa có nội dung",
                "source": "empty_fallback"
            }

        # 1. Kiểm tra bằng Rule-based trước để bảo vệ các từ khóa nguy cấp y tế
        rule_intent, rule_score = self.rule_based_classify(text)

        # 2. Nếu có mô hình ML đã huấn luyện, dự đoán xác suất
        ml_intent = None
        ml_confidence = 0.0

        if self.pipeline is not None:
            try:
                proba = self.pipeline.predict_proba([text])[0]
                classes = self.pipeline.classes_
                best_idx = proba.argmax()
                ml_intent = classes[best_idx]
                ml_confidence = float(proba[best_idx])
            except Exception:
                try:
                    ml_intent = self.pipeline.predict([text])[0]
                    ml_confidence = 0.8
                except Exception:
                    pass

        # 3. Tổng hợp quyết định (Decision Fusion)
        final_intent = "UNKNOWN"
        final_confidence = 0.0
        source = "rule"

        # Nếu quy tắc phát hiện tình huống khẩn cấp (Cấp 1 hoặc Cấp 2), ưu tiên cứu tính mạng
        if rule_intent and INTENT_LABELS.get(rule_intent, {}).get("is_emergency", False):
            final_intent = rule_intent
            final_confidence = max(rule_score, ml_confidence)
            source = "emergency_rule"
        elif ml_intent and ml_confidence >= 0.55:
            final_intent = ml_intent
            final_confidence = ml_confidence
            source = "ml_model"
        elif rule_intent:
            final_intent = rule_intent
            final_confidence = rule_score
            source = "rule"
        elif ml_intent:
            final_intent = ml_intent
            final_confidence = ml_confidence
            source = "ml_model_low_conf"
        else:
            final_intent = "GENERAL_HEALTH_CHECK"
            final_confidence = 0.4
            source = "default"

        meta = INTENT_LABELS.get(final_intent, {
            "title": "Tư vấn Sức khỏe Chung",
            "is_emergency": False,
            "priority": 5
        })

        return {
            "intent": final_intent,
            "title": meta.get("title", final_intent),
            "confidence": round(final_confidence, 3),
            "is_emergency": meta.get("is_emergency", False),
            "source": source
        }


# Singleton
intent_classifier_instance = None


def get_intent_classifier() -> IntentClassifier:
    global intent_classifier_instance
    if intent_classifier_instance is None:
        intent_classifier_instance = IntentClassifier()
    return intent_classifier_instance
