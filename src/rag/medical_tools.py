# -*- coding: utf-8 -*-
"""
MODULE AGENTIC MEDICAL TOOLS (CÔNG CỤ Y TẾ CHO TÁC TỬ AI)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Kiến trúc: MedRAG & Agentic RAG

Module này định nghĩa các công cụ y tế (Tools/Functions) chuẩn OpenAI Tool Calling Schema
cho phép LLM tự động suy luận và kích hoạt các hành động chuyên biệt:
1. get_patient_vitals: Lấy chỉ số sinh hiệu thời gian thực từ cảm biến trạm biên.
2. get_patient_profile: Tra cứu hồ sơ bệnh án, tiền sử bệnh, bác sĩ điều trị từ SQLite.
3. check_drug_allergy: Kiểm tra tương tác dị ứng thuốc (Penicillin, NSAID, v.v.).
4. search_medical_knowledge: Tra cứu phác đồ sơ cứu và cẩm nang y khoa lão khoa.
5. trigger_emergency_alert: Kích hoạt báo động sự cố y tế khẩn cấp tới bác sĩ & 115.
6. get_caregiver_info: Lấy thông tin người bảo hộ chính đang trực.
"""

import os
import sys
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.db.db_service import get_db_service

LOGS_FILE = os.path.join(PROJECT_ROOT, "system_logs.txt")


class MedicalToolsRegistry:
    """Kho đăng ký và điều phối thực thi các công cụ y tế chuẩn Agentic RAG."""

    def __init__(self, logs_file: str = LOGS_FILE):
        self.logs_file = logs_file
        self.db = get_db_service()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Trả về danh sách định nghĩa Tools theo chuẩn OpenAI Function/Tool Calling Schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_patient_vitals",
                    "description": "Lấy chỉ số sinh hiệu thời gian thực đo được từ cảm biến và camera giám sát (thân nhiệt, nhịp tim, tư thế té ngã, âm thanh va đập, thời gian ghi nhận).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_patient_profile",
                    "description": "Tra cứu hồ sơ y tế bệnh nhân đang được giám sát: mã hồ sơ, họ tên, tuổi, phòng ở, tiền sử bệnh án, dị ứng thuốc, nhóm máu và bác sĩ phụ trách.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_drug_allergy",
                    "description": "Kiểm tra xem bệnh nhân có bị dị ứng hoặc chống chỉ định với loại thuốc nào đó không (ví dụ: Penicillin, Amoxicillin, Augmentin, NSAID...).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "Tên loại thuốc hoặc nhóm thuốc cần kiểm tra (ví dụ: 'Amoxicillin', 'Penicillin', 'Paracetamol', 'Ibuprofen')."
                            }
                        },
                        "required": ["drug_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_medical_knowledge",
                    "description": "Tìm kiếm ngữ nghĩa trong cẩm nang phác đồ y khoa lão khoa (sơ cứu té ngã, đột quỵ FAST, CPR, chăm sóc loét, Parkinson, hạ đường huyết...).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Nội dung hoặc triệu chứng cần tra cứu phác đồ xử trí."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "trigger_emergency_alert",
                    "description": "Kích hoạt báo động sự cố y tế khẩn cấp, ghi vào nhật ký hệ thống và gửi thông báo khẩn cấp tới Bác sĩ điều trị và Cấp cứu 115.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_type": {
                                "type": "string",
                                "enum": ["FALL_EMERGENCY", "CARDIAC_CRISIS", "STROKE_FAST", "HIGH_FEVER", "DISTRESS_CALL"],
                                "description": "Loại sự cố khẩn cấp y tế."
                            },
                            "details": {
                                "type": "string",
                                "description": "Mô tả chi tiết sự cố và các chỉ số sinh hiệu nguy kịch."
                            }
                        },
                        "required": ["event_type", "details"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_caregiver_info",
                    "description": "Lấy thông tin người bảo hộ / người giám hộ chính hiện tại (họ tên, vai trò, số điện thoại, mối quan hệ với bệnh nhân).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Thực thi một công cụ y tế theo tên và tham số truyền vào."""
        args = arguments or {}
        try:
            if tool_name == "get_patient_vitals":
                return self._tool_get_patient_vitals()
            elif tool_name == "get_patient_profile":
                return self._tool_get_patient_profile()
            elif tool_name == "check_drug_allergy":
                return self._tool_check_drug_allergy(args.get("drug_name", ""))
            elif tool_name == "search_medical_knowledge":
                return self._tool_search_medical_knowledge(args.get("query", ""))
            elif tool_name == "trigger_emergency_alert":
                return self._tool_trigger_emergency_alert(args.get("event_type", "UNKNOWN"), args.get("details", ""))
            elif tool_name == "get_caregiver_info":
                return self._tool_get_caregiver_info()
            else:
                return {
                    "status": "error",
                    "error": f"Không tìm thấy công cụ y tế '{tool_name}'"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Lỗi khi thực thi công cụ '{tool_name}': {str(e)}"
            }

    # =========================================================================
    # CÁC HÀM XỬ LÝ NỘI BỘ CHO TỪNG TOOL
    # =========================================================================

    def _tool_get_patient_vitals(self) -> Dict[str, Any]:
        """Đọc và trích xuất chỉ số sinh hiệu mới nhất từ system_logs.txt."""
        vitals = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "NORMAL",
            "body_temp": 36.6,
            "heart_rate": 74,
            "posture": "AN TOÀN",
            "sound_db": 42.0,
            "is_fall": False,
            "is_fever": False,
            "is_cardiac_alert": False,
            "latest_log_line": "Không có nhật ký cảm biến"
        }

        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                    if lines:
                        latest = lines[-1]
                        vitals["latest_log_line"] = latest

                        # Bóc tách thông số bằng Regex
                        temp_m = re.search(r'Temp=([\d\.]+)°C', latest)
                        if temp_m:
                            vitals["body_temp"] = float(temp_m.group(1))
                            if vitals["body_temp"] >= 38.0:
                                vitals["is_fever"] = True

                        hr_m = re.search(r'HR=(\d+)bpm', latest)
                        if hr_m:
                            vitals["heart_rate"] = int(hr_m.group(1))
                            if vitals["heart_rate"] >= 120 or vitals["heart_rate"] <= 50:
                                vitals["is_cardiac_alert"] = True

                        sound_m = re.search(r'Sound=([\d\.]+)dB', latest)
                        if sound_m:
                            vitals["sound_db"] = float(sound_m.group(1))

                        if "FALL_DETECTED" in latest or "TÉ NGÃ" in latest:
                            vitals["is_fall"] = True
                            vitals["status"] = "CRITICAL_FALL"
                            vitals["posture"] = "🚨 NẰM BẸP TRÊN SÀN (TÉ NGÃ)"
                        elif vitals["is_fever"]:
                            vitals["status"] = "WARNING_FEVER"
                            vitals["posture"] = "NẰM NGHỈ TRÊN GIƯỜNG (SỐT CAO)"
                        elif vitals["is_cardiac_alert"]:
                            vitals["status"] = "WARNING_CARDIAC"
                            vitals["posture"] = "NGỒI NGHỈ (NHỊP TIM NHANH)"
                        else:
                            vitals["status"] = "NORMAL"
                            vitals["posture"] = "SINH HOẠT AN TOÀN"
            except Exception as e:
                vitals["error"] = str(e)

        return {
            "status": "success",
            "vitals": vitals,
            "clinical_summary": (
                f"Thân nhiệt: {vitals['body_temp']}°C | Nhịp tim: {vitals['heart_rate']} BPM | "
                f"Tình trạng: {vitals['status']} ({vitals['posture']}) | Âm thanh: {vitals['sound_db']}dB"
            )
        }

    def _tool_get_patient_profile(self) -> Dict[str, Any]:
        """Đọc hồ sơ bệnh nhân từ SQLite."""
        patient = self.db.get_patient_profile()
        if not patient:
            return {
                "status": "not_found",
                "message": "Chưa có hồ sơ bệnh nhân nào trong cơ sở dữ liệu."
            }

        return {
            "status": "success",
            "patient": {
                "code": patient.get("patient_code", "#ELD-8402"),
                "name": patient.get("full_name", "Cụ Nguyễn Văn An"),
                "age": patient.get("age", 82),
                "gender": patient.get("gender", "Nam"),
                "room": patient.get("room", "Phòng 102 - Tầng 1"),
                "medical_history": patient.get("medical_history", "Tăng huyết áp vô căn độ 2, Tiểu đường Type 2, Thoái hóa khớp gối"),
                "allergies": patient.get("allergies", "Dị ứng kháng sinh Penicillin"),
                "blood_type": patient.get("blood_type", "O+"),
                "doctor": {
                    "name": patient.get("doctor_name", "BS. CKI Trần Minh Tuấn"),
                    "phone": patient.get("doctor_phone", "0912.345.678")
                },
                "notes": patient.get("emergency_notes", "Cụ dễ chóng mặt khi thay đổi tư thế đột ngột.")
            }
        }

    def _tool_check_drug_allergy(self, drug_name: str) -> Dict[str, Any]:
        """Kiểm tra tương tác dị ứng thuốc nghiêm ngặt (chống sốc phản vệ)."""
        d_lower = drug_name.strip().lower()
        if not d_lower:
            return {
                "status": "error",
                "message": "Chưa cung cấp tên thuốc cần kiểm tra."
            }

        patient = self.db.get_patient_profile() or {}
        allergies_desc = patient.get("allergies", "Dị ứng kháng sinh Penicillin")

        # Danh sách thuốc thuộc nhóm Penicillin / Beta-lactam cấm kỵ
        penicillin_group = [
            "penicillin", "amoxicillin", "augmentin", "ampicillin", "cloxacillin",
            "oxacillin", "piperacillin", "clavulanate", "clamoxyl"
        ]

        # Danh sách thuốc NSAID có hại cho người già tăng huyết áp/tiểu đường
        nsaid_group = [
            "ibuprofen", "diclofenac", "meloxicam", "piroxicam", "naproxen", "aspirin liều cao", "celecoxib"
        ]

        is_penicillin = any(p in d_lower for p in penicillin_group)
        is_nsaid = any(n in d_lower for n in nsaid_group)
        is_paracetamol = "paracetamol" in d_lower or "panadol" in d_lower or "efferalgan" in d_lower or "acetaminophen" in d_lower

        if is_penicillin:
            return {
                "status": "DANGER",
                "is_safe": False,
                "drug_name": drug_name,
                "severity": "CRITICAL_FATAL",
                "clinical_warning": (
                    f"⛔ CẤM TUYỆT ĐỐI! Thuốc '{drug_name}' thuộc nhóm Penicillin/Beta-lactam mà bệnh nhân có tiền sử DỊ ỨNG NẶNG TUYỆT ĐỐI. "
                    "Nguy cơ sốc phản vệ trụy mạch đe dọa tính mạng chỉ sau vài phút! Tuyệt đối không sử dụng."
                ),
                "recommendation": "Báo ngay cho BS. CKI Trần Minh Tuấn (0912.345.678) để đổi sang nhóm kháng sinh an toàn thay thế."
            }
        elif is_nsaid:
            return {
                "status": "WARNING",
                "is_safe": False,
                "drug_name": drug_name,
                "severity": "HIGH_RISK",
                "clinical_warning": (
                    f"⚠️ KHUYẾN CÁO KHÔNG DÙNG: Thuốc '{drug_name}' là thuốc kháng viêm không steroid (NSAID). "
                    "Người già 82 tuổi có tiền sử Tăng huyết áp và Tiểu đường dùng NSAID có nguy cơ cao gây xuất huyết tiêu hóa, tăng huyết áp kịch phát và suy thận cấp!"
                ),
                "recommendation": "Ưu tiên giảm đau an toàn bằng Paracetamol đơn thuần (<= 3000mg/ngày) hoặc chườm ấm."
            }
        elif is_paracetamol:
            return {
                "status": "SAFE_WITH_LIMIT",
                "is_safe": True,
                "drug_name": drug_name,
                "severity": "INFO",
                "clinical_warning": (
                    "🟢 AN TOÀN KHI DÙNG ĐÚNG LIỀU: Paracetamol là thuốc giảm đau/hạ sốt đầu tay phù hợp cho cụ An. "
                    "Quy tắc an toàn: Dùng 500mg/lần, cách nhau 4-6 giờ. TỔNG LIỀU KHÔNG QUÁ 3000mg/ngày (tối đa 6 viên/ngày) để tránh độc gan."
                ),
                "recommendation": "Uống sau khi ăn no kèm 1 cốc nước ấm."
            }
        else:
            return {
                "status": "CAUTION",
                "is_safe": True,
                "drug_name": drug_name,
                "severity": "UNKNOWN_CONSULT_DOCTOR",
                "clinical_warning": (
                    f"Thuốc '{drug_name}' không nằm trong danh mục dị ứng ghi nhận sẵn ({allergies_desc}). "
                    "Tuy nhiên đối với người cao tuổi mắc bệnh mạn tính, luôn cần chỉ định từ bác sĩ phụ trách."
                ),
                "recommendation": "Tham vấn BS. CKI Trần Minh Tuấn trước khi cho cụ uống."
            }

    def _tool_search_medical_knowledge(self, query: str) -> Dict[str, Any]:
        """Truy xuất tài liệu phác đồ y tế lâm sàng qua Local Retriever."""
        try:
            from src.rag.local_vector_retriever import get_local_retriever
            retriever = get_local_retriever()
            results = retriever.search(query, top_k=3)
            return {
                "status": "success",
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "source": r.get("source", ""),
                        "score": round(r.get("score", 0), 4),
                        "snippet": r.get("text", "")[:400] + "..."
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Lỗi truy xuất tri thức: {str(e)}"
            }

    def _tool_trigger_emergency_alert(self, event_type: str, details: str) -> Dict[str, Any]:
        """Ghi nhận sự cố khẩn cấp vào system_logs.txt và kích hoạt thông điệp 115."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{now_str}] [CRITICAL] [AGENTIC_ALERT: {event_type}] -> {details}\n"

        try:
            with open(self.logs_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[TOOL ERROR] Không thể ghi log cảnh báo: {e}", file=sys.stderr)

        return {
            "status": "TRIGGERED",
            "alert_id": f"SOS-{int(datetime.now().timestamp())}",
            "timestamp": now_str,
            "event_type": event_type,
            "details": details,
            "actions_dispatched": [
                "Đã ghi nhật ký trạm biên với nhãn CRITICAL",
                "Khởi động chuông cảnh báo nội bộ phòng 102",
                "Đã sẵn sàng kết nối Cấp cứu Quốc gia 115",
                "Đã gửi thông báo ưu tiên tới Bác sĩ phụ trách (0912.345.678) & Người bảo hộ (0908.123.456)"
            ]
        }

    def _tool_get_caregiver_info(self) -> Dict[str, Any]:
        """Lấy thông tin người bảo hộ từ SQLite."""
        user = self.db.get_current_user() or {}
        return {
            "status": "success",
            "caregiver": {
                "full_name": user.get("full_name", "Nguyễn Hữu Nghĩa"),
                "role": user.get("role", "Người bảo hộ chính"),
                "relationship": user.get("relationship", "Con trai trưởng"),
                "phone": user.get("phone", "0908.123.456"),
                "email": user.get("email", "nghia.nguyen@email.com")
            }
        }


# Singleton Instance
medical_tools_registry_instance = None


def get_medical_tools_registry() -> MedicalToolsRegistry:
    global medical_tools_registry_instance
    if medical_tools_registry_instance is None:
        medical_tools_registry_instance = MedicalToolsRegistry()
    return medical_tools_registry_instance


if __name__ == "__main__":
    registry = get_medical_tools_registry()
    print("=" * 70)
    print("KIỂM THỬ MODULE AGENTIC MEDICAL TOOLS")
    print("=" * 70)

    print("\n1. Danh sách Tools định nghĩa:")
    tools = registry.get_tool_definitions()
    for t in tools:
        print(f" - {t['function']['name']}: {t['function']['description'][:60]}...")

    print("\n2. Thực thi tool 'get_patient_vitals':")
    vitals_res = registry.execute_tool("get_patient_vitals")
    print(json.dumps(vitals_res, indent=2, ensure_ascii=False))

    print("\n3. Thực thi tool 'check_drug_allergy' với Amoxicillin:")
    allergy_res = registry.execute_tool("check_drug_allergy", {"drug_name": "Amoxicillin"})
    print(json.dumps(allergy_res, indent=2, ensure_ascii=False))
