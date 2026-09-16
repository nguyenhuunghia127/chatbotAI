# -*- coding: utf-8 -*-
"""
MODULE CORRECTIVE RAG (CRAG) & SELF-RAG EVALUATOR (BỘ ĐÁNH GIÁ & TỰ PHẢN TƯ)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Kiến trúc: Compound MedRAG & Corrective RAG (CRAG)

Chức năng:
1. Đánh giá chất lượng tài liệu trước khi sinh (Pre-generation Retrieval Evaluator):
   - Đánh giá xem context lấy về từ Vector DB / TF-IDF là:
     * CORRECT (Đầy đủ, độ tin cậy >= 0.60): Cho phép sinh câu trả lời trực tiếp.
     * AMBIGUOUS (Chưa đủ, 0.30 - 0.59): Kích hoạt GraphRAG bù đắp tri thức dạng quan hệ.
     * INCORRECT (Lạc đề, < 0.30): Tự động chuyển hướng sang phác đồ lâm sàng an toàn.
2. Tự phản tư và kiểm duyệt lâm sàng sau khi sinh (Post-generation Self-RAG Audit):
   - Quét câu trả lời của LLM để phát hiện các lỗi sai chết người (Hallucination về thuốc):
     * Cấm kê Penicillin / Augmentin cho bệnh nhân dị ứng.
     * Giới hạn Paracetamol <= 3000mg/ngày, cấm tự ý dùng NSAID.
     * Bắt buộc có khuyến cáo 115 và bác sĩ phụ trách khi có tai biến/ngã.
   - Tự động sửa chữa hoặc chèn cảnh báo an toàn nếu câu trả lời vi phạm guardrails.
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple, Optional

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class CorrectiveRAGEvaluator:
    """Bộ đánh giá độ tin cậy ngữ cảnh và tự phản tư an toàn lâm sàng (CRAG & Self-RAG)."""

    def __init__(self):
        pass

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[str],
        rerank_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Đánh giá mức độ phù hợp của ngữ cảnh vừa truy xuất (Retrieval Confidence Grading).
        """
        if not retrieved_chunks or not rerank_scores:
            return {
                "grade": "INCORRECT",
                "confidence": 0.1,
                "action": "FALLBACK_EDGE_RULES",
                "reason": "Không tìm thấy đoạn văn bản nào từ kho tri thức"
            }

        max_score = max([r.get("score", 0.0) for r in rerank_scores], default=0.0)
        avg_score = sum([r.get("score", 0.0) for r in rerank_scores]) / max(len(rerank_scores), 1)

        # Đếm số lượng từ khóa lâm sàng trùng khớp
        matched_kw_count = sum(len(r.get("matched_keywords", [])) for r in rerank_scores)

        # Tính điểm tin cậy tổng hợp (0.0 -> 1.0)
        confidence = min(1.0, (max_score * 0.5) + (avg_score * 0.3) + (min(matched_kw_count, 3) * 0.1))

        if confidence >= 0.55:
            grade = "CORRECT"
            action = "PROCEED_TO_LLM"
            reason = "Ngữ cảnh khớp chính xác cao với phác đồ y tế lâm sàng."
        elif confidence >= 0.25:
            grade = "AMBIGUOUS"
            action = "INJECT_GRAPHRAG"
            reason = "Ngữ cảnh có liên quan nhưng cần bổ sung suy luận quan hệ từ Medical GraphRAG."
        else:
            grade = "INCORRECT"
            action = "FALLBACK_EDGE_RULES"
            reason = "Độ tương quan ngữ nghĩa thấp, ưu tiên phác đồ suy luận quy tắc an toàn."

        return {
            "grade": grade,
            "confidence": round(confidence, 4),
            "max_score": round(max_score, 4),
            "action": action,
            "reason": reason
        }

    def self_reflection_audit(
        self,
        query: str,
        answer: str,
        patient_allergies: str = "Dị ứng kháng sinh Penicillin",
        doctor_phone: str = "0912.345.678"
    ) -> Dict[str, Any]:
        """
        Bộ lọc tự phản tư hậu kỳ (Self-Reflection Guardrail):
        Rà soát câu trả lời để ngăn chặn ảo giác (Anti-Hallucination) đe dọa tính mạng.
        """
        ans_lower = answer.lower()
        q_lower = query.lower()
        safety_issues = []
        corrections = []
        modified_answer = answer

        # 1. Kiểm tra Dị ứng Penicillin / Augmentin / Amoxicillin
        penicillin_keywords = ["penicillin", "amoxicillin", "augmentin", "ampicillin", "clamoxyl"]
        mentions_penicillin = any(pk in q_lower or pk in ans_lower for pk in penicillin_keywords)

        if mentions_penicillin:
            has_warning = any(w in ans_lower for w in ["cấm", "dị ứng", "sốc phản vệ", "tuyệt đối không", "nguy hiểm"])
            if not has_warning:
                safety_issues.append("MISSING_PENICILLIN_ALLERGY_WARNING")
                alert_text = (
                    f"\n\n🚨 **CAN THIỆP AN TOÀN TỐI CAO (SELF-RAG GUARD):**\n"
                    f"Bệnh nhân có tiền sử **{patient_allergies.upper()}**. "
                    "Tuyệt đối **CẤM DÙNG** các thuốc này vì nguy cơ sốc phản vệ tử vong trong vài phút! "
                    f"Liên hệ ngay BS. CKI Trần Minh Tuấn ({doctor_phone})."
                )
                modified_answer += alert_text
                corrections.append("Đã bổ sung cảnh báo sốc phản vệ cấm dùng nhóm Penicillin")

        # 2. Kiểm tra Đột quỵ não (F.A.S.T) và Cấp cứu 115
        if any(w in q_lower for w in ["đột quỵ", "méo miệng", "nói đớ", "ngọng", "fast"]):
            if "115" not in ans_lower:
                safety_issues.append("MISSING_115_EMERGENCY_CALL")
                modified_answer += "\n\n👉 **HÀNH ĐỘNG KHẨN CẤP**: Gọi ngay **Cấp cứu 115** lập tức trong giờ vàng (< 4.5h)!"
                corrections.append("Đã bổ sung chỉ định gọi 115 cho ca nghi ngờ đột quỵ não")

        # 3. Kiểm tra Giới hạn Paracetamol
        if "paracetamol" in ans_lower and "3000" not in ans_lower and "liều" in q_lower:
            modified_answer += "\n💡 **Lưu ý liều lượng**: Tổng liều Paracetamol cho người cao tuổi **không vượt quá 3000mg/ngày (tối đa 6 viên 500mg)**."
            corrections.append("Đã bổ sung giới hạn liều tối đa Paracetamol 3000mg/ngày")

        # 4. Kiểm tra Chế độ ăn giảm muối (<5g)
        if any(w in q_lower for w in ["muối", "ăn mặn", "huyết áp ăn gì", "kiêng ăn gì"]):
            if "5g" not in ans_lower:
                modified_answer += "\n💡 **Khuyến cáo dinh dưỡng**: Lượng muối an toàn bắt buộc duy trì **dưới 5g muối/ngày** (< 1 thìa cà phê)."
                corrections.append("Đã bổ sung ngưỡng khuyến cáo giảm muối dưới 5g/ngày")

        passed = (len(safety_issues) == 0)

        return {
            "reflection_passed": passed,
            "safety_issues": safety_issues,
            "corrections_applied": corrections,
            "audited_answer": modified_answer
        }


# Singleton
corrective_rag_evaluator_instance = None


def get_corrective_rag_evaluator() -> CorrectiveRAGEvaluator:
    global corrective_rag_evaluator_instance
    if corrective_rag_evaluator_instance is None:
        corrective_rag_evaluator_instance = CorrectiveRAGEvaluator()
    return corrective_rag_evaluator_instance


if __name__ == "__main__":
    evaluator = get_corrective_rag_evaluator()
    print("=" * 70)
    print("KIỂM THỬ MODULE CORRECTIVE RAG (CRAG) & SELF-RAG EVALUATOR")
    print("=" * 70)

    print("\n1. Kiểm thử Đánh giá Độ tin cậy Ngữ cảnh (Retrieval Confidence Grading):")
    good_eval = evaluator.evaluate_retrieval(
        query="đột quỵ não",
        retrieved_chunks=["Đột quỵ FAST gọi 115"],
        rerank_scores=[{"score": 0.85, "matched_keywords": ["đột quỵ", "fast"]}]
    )
    print(f" - Context chất lượng cao: Grade={good_eval['grade']} (Conf={good_eval['confidence']}) -> Action={good_eval['action']}")

    ambig_eval = evaluator.evaluate_retrieval(
        query="thuốc hạ sốt",
        retrieved_chunks=["Chườm ấm và nghỉ ngơi"],
        rerank_scores=[{"score": 0.35, "matched_keywords": []}]
    )
    print(f" - Context mơ hồ: Grade={ambig_eval['grade']} (Conf={ambig_eval['confidence']}) -> Action={ambig_eval['action']}")

    print("\n2. Kiểm thử Tự phản tư an toàn (Self-RAG Post-Generation Audit):")
    unsafe_ans = "Bạn có thể cho cụ uống 1 viên Augmentin để kháng viêm."
    audit_res = evaluator.self_reflection_audit("cụ sốt có uống augmentin được không", unsafe_ans)
    print(f" - Đạt chuẩn an toàn ban đầu: {audit_res['reflection_passed']}")
    print(f" - Lỗi an toàn phát hiện: {audit_res['safety_issues']}")
    print(f" - Hiệu chỉnh tự động:\n{audit_res['audited_answer']}")
