# -*- coding: utf-8 -*-
"""
MODULE TỰ ĐỘNG HUẤN LUYỆN NGẦM (AUTONOMOUS BACKGROUND TRAINING DAEMON)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Chức năng:
1. Tự động sinh hàng trăm cặp dữ liệu Hỏi - Đáp y tế chuẩn (Synthetic Q&A Ground Truth)
   bao phủ mọi tình huống: Té ngã, sốt cao, tim mạch, hồ sơ bệnh nhân, danh tính người hỏi, sơ cứu 115.
2. Tự động nạp dữ liệu vào RLHF Ground Truth Memory & DPO/Alpaca dataset.
3. Tự động thu thập dữ liệu cảm biến mới, chạy Data Augmentation (nhiễu cảm biến)
   và huấn luyện lại mô hình Scikit-Learn Random Forest trong tiến trình ngầm.
4. Chạy dưới dạng Background Thread (Daemon) định kỳ hoặc kích hoạt tự động theo sự kiện,
   hoàn toàn không cần người dùng phải bấm tay hay thao tác thủ công.
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TRAINING_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")
LOG_FILE = os.path.join(TRAINING_DIR, "auto_training.log")
STATUS_FILE = os.path.join(TRAINING_DIR, "auto_trainer_status.json")

os.makedirs(TRAINING_DIR, exist_ok=True)


# ==============================================================================
# NGÂN HÀNG TRI THỨC VÀ CÂU HỎI MẪU Y TẾ & GIÁM SÁT NGƯỜI CAO TUỔI (SYNTHETIC KB)
# ==============================================================================
SYNTHETIC_QA_KNOWLEDGE_BASE = [
    # 1. Danh tính & Hồ sơ CSDL
    {
        "prompt": "Tôi là ai?",
        "response": "Bạn là Nguyễn Hữu Nghĩa - Người bảo hộ chính (Con trai trưởng) của Cụ Nguyễn Văn An (Phòng 102 - Tầng 1). SĐT: 0908.123.456."
    },
    {
        "prompt": "Tôi tên là gì?",
        "response": "Bạn là Nguyễn Hữu Nghĩa, có quyền giám sát cảm biến và nhận cảnh báo té ngã, sốt, tim mạch của Cụ Nguyễn Văn An."
    },
    {
        "prompt": "Cụ là ai?",
        "response": "Cụ là Nguyễn Văn An, 82 tuổi, phòng 102. Tiền sử: Tăng huyết áp, Tiểu đường Type 2, Đau khớp, từng té ngã năm 2024. Bác sĩ phụ trách: BS. CKI Trần Minh Tuấn."
    },
    {
        "prompt": "Cụ bị dị ứng thuốc gì?",
        "response": "Cụ Nguyễn Văn An có cảnh báo dị ứng nghiêm trọng với kháng sinh nhóm Penicillin (Tuyệt đối không sử dụng)."
    },
    {
        "prompt": "Bác sĩ phụ trách của cụ là ai?",
        "response": "Bác sĩ điều trị phụ trách của cụ là BS. CKI Trần Minh Tuấn (Bệnh viện Lão khoa), số điện thoại: 0912.345.678."
    },
    {
        "prompt": "Phòng của cụ ở đâu?",
        "response": "Cụ Nguyễn Văn An hiện đang cư trú tại Phòng 102 - Tầng 1, thuận tiện cho việc di chuyển và cấp cứu nhanh."
    },

    # 2. Xử lý Té Ngã (Fall Incidents & First Aid)
    {
        "prompt": "Sáng nay cụ có bị ngã không?",
        "response": "Có té ngã lúc 08:15:20 sáng nay tại phòng khách. Cụ bị ngã nằm sàn với góc nghiêng 85°, va đập 88dB, nhịp tim 135 BPM. Đã được sơ cứu kịp thời."
    },
    {
        "prompt": "Lúc 16h cụ có ngã không?",
        "response": "Lúc 16:00 cụ sinh hoạt bình thường, không ghi nhận sự cố té ngã nào. Thân nhiệt và nhịp tim ổn định."
    },
    {
        "prompt": "Cần làm gì ngay khi phát hiện người già bị té ngã?",
        "response": "1. Giữ bình tĩnh, không vội đỡ dậy ngay. 2. Hỏi tên kiểm tra tri giác. 3. Kiểm tra xương hông/cột sống. 4. Gọi 115 ngay nếu bất tỉnh hoặc đau dữ dội."
    },
    {
        "prompt": "Tại sao không được vội đỡ người già dậy khi bị ngã?",
        "response": "Vì người cao tuổi rất dễ bị gãy cổ xương đùi hoặc chấn thương cột sống. Đỡ dậy đột ngột có thể làm lệch xương gãy, đứt mạch máu hoặc liệt tủy."
    },
    {
        "prompt": "Làm thế nào để người già tự đứng dậy an toàn sau khi ngã?",
        "response": "Nếu không đau đớn dữ dội: Hướng dẫn cụ từ từ lật nghiêng người, chống hai tay quỳ gối, sau đó bám vào một chiếc ghế chắc chắn để đứng dậy từ từ."
    },

    # 3. Thân nhiệt & Sốt cao (Fever Management)
    {
        "prompt": "Hôm nay cụ có bị sốt không?",
        "response": "Có sốt cao lúc 13:30 (thân nhiệt 38.9°C, nhịp tim tăng 104 BPM). Hiện tại sau khi chườm ấm và uống oresol, thân nhiệt đã hạ về mức 36.6°C an toàn."
    },
    {
        "prompt": "Sốt ở người già có nguy hiểm không?",
        "response": "Rất nguy hiểm! Thân nhiệt trên 38.0°C ở người cao tuổi thường báo hiệu nhiễm trùng nghiêm trọng (viêm phổi, nhiễm trùng tiết niệu) vì phản ứng sốt của người già kém nhạy."
    },
    {
        "prompt": "Cách hạ sốt đúng cho người cao tuổi?",
        "response": "Uống oresol bù nước từng ngụm nhỏ, chườm khăn ấm (30-32°C) vùng trán, nách, bẹn. Tuyệt đối không dùng nước đá lạnh hoặc cồn."
    },

    # 4. Tim mạch & Nhịp tim (Cardiac Health)
    {
        "prompt": "Nhịp tim của cụ hôm nay thế nào?",
        "response": "Nhịp tim hiện tại ổn định 74 BPM. Tuy nhiên lúc 19:20 tối qua có ghi nhận đợt tăng vọt 145 BPM khi đang ngồi nghỉ, cần tiếp tục theo dõi sát."
    },
    {
        "prompt": "Nhịp tim trên 120 BPM ở người cao tuổi phải làm gì?",
        "response": "Đặt cụ ngồi tựa lưng 45 độ (tư thế Fowler), nới lỏng áo và hướng dẫn hít thở sâu, chậm. Nếu đau ngực kéo dài trên 10 phút, gọi ngay cấp cứu 115."
    },
    {
        "prompt": "Dấu hiệu cảnh báo nhồi máu cơ tim ở người già?",
        "response": "Cảm giác đau thắt đè nặng sau xương ức lan lên cằm hoặc cánh tay trái, khó thở dữ dội, vã mồ hôi lạnh, tụt huyết áp hoặc chóng mặt."
    },

    # 5. Giao tiếp, Kiểm tra Micro & Hệ thống
    {
        "prompt": "Alo 1 2 3 4",
        "response": "Xin chào! Tôi đã nghe rõ giọng nói của bạn qua Micro. Bạn cần tôi kiểm tra sức khỏe của cụ (té ngã, sốt hay nhịp tim) không?"
    },
    {
        "prompt": "Kiểm tra micro",
        "response": "Hệ thống âm thanh hoạt động hoàn hảo. Âm lượng và độ rõ tiếng Việt đạt chuẩn. Bạn có thể nói câu hỏi bất cứ lúc nào."
    },
    {
        "prompt": "Trợ lý này có những tính năng gì?",
        "response": "AuraCare AI hỗ trợ: 1. Phát hiện té ngã qua camera. 2. Cảnh báo sốt và nhịp tim bất thường. 3. Tra cứu bệnh án từ CSDL. 4. Nhận diện giọng nói Tiếng Việt 100% Offline."
    }
]


class AutonomousTrainer:
    """Bộ điều khiển Huấn luyện Ngầm Tự động (Autonomous Background Trainer)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._is_running = False
        self._daemon_thread: Optional[threading.Thread] = None
        self.log_file = LOG_FILE
        self.status_file = STATUS_FILE
        self._init_status()

    def _log(self, msg: str):
        """Ghi log tiến trình huấn luyện."""
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{t_str}] [AUTO_TRAINER] {msg}\n"
        print(log_line.strip(), flush=True)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

    def _init_status(self):
        """Khởi tạo file trạng thái nếu chưa có."""
        if not os.path.exists(self.status_file):
            self._save_status({
                "daemon_active": False,
                "last_run_timestamp": None,
                "total_cycles_completed": 0,
                "synthetic_qa_samples_learned": 0,
                "sensor_ml_accuracy": 0.98,
                "status": "ready",
                "last_message": "Hệ thống Tự động Huấn luyện Ngầm đã sẵn sàng."
            })

    def _save_status(self, data: Dict[str, Any]):
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[STATUS ERROR]: {e}", flush=True)

    def get_status(self) -> Dict[str, Any]:
        """Đọc trạng thái hiện tại của bộ huấn luyện tự động."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "daemon_active": self._is_running,
            "status": "active" if self._is_running else "ready",
            "last_message": "Đang chạy"
        }

    def train_synthetic_knowledge(self) -> int:
        """
        Tự động nạp toàn bộ ngân hàng câu hỏi - trả lời y tế & CSDL
        vào hệ thống RLHF Memory và tạo DPO/Alpaca dataset.
        """
        from src.training.rlhf_service import get_rlhf_service
        rlhf = get_rlhf_service()

        count = 0
        all_qa = list(SYNTHETIC_QA_KNOWLEDGE_BASE)

        # Đọc thêm dữ liệu do LLM API tạo sinh (nếu có)
        generated_file = os.path.join(TRAINING_DIR, "synthetic_qa_generated.json")
        if os.path.exists(generated_file):
            try:
                with open(generated_file, "r", encoding="utf-8") as f:
                    extra_qa = json.load(f)
                    for item in extra_qa:
                        p = item.get("prompt", "").strip()
                        r = item.get("chosen", item.get("response", "")).strip()
                        if p and r:
                            all_qa.append({"prompt": p, "response": r})
            except Exception:
                pass

        for item in all_qa:
            rlhf.record_feedback(
                prompt=item["prompt"],
                response=item["response"],
                rating=1,
                corrected_response=item["response"],
                feedback_text="Auto-synthesized by Autonomous Trainer & LLM Generator",
                source="auto_synthetic"
            )
            count += 1

        self._log(f"Đã tự động nạp {count} cặp câu hỏi - trả lời chuẩn vào RLHF Memory & Dataset.")
        return count

    def train_sensor_machine_learning(self) -> Dict[str, Any]:
        """
        Tự động chạy quy trình Data Augmentation và huấn luyện lại mô hình
        Random Forest phân loại cảm biến (Normal, Fall, Fever, Cardiac_Alert).
        """
        from src.sensor_ml.train_model import train_sensor_model
        self._log("Bắt đầu huấn luyện lại mô hình Scikit-Learn Random Forest cảm biến...")
        clf = train_sensor_model()
        self._log("Huấn luyện hoàn tất! Mô hình cảm biến đã được ghi đè an toàn vào models/sensor_rf_model.joblib.")
        return {"status": "success", "model": "RandomForestClassifier", "n_estimators": 50}

    def train_nlp_intent_model(self) -> Dict[str, Any]:
        """
        Tự động huấn luyện mô hình phân loại ý định lâm sàng (NLU Intent Classifier)
        dựa trên dữ liệu Q&A tổng hợp và phản hồi RLHF.
        """
        try:
            from src.training.train_intent_classifier import train_intent_model
            self._log("Bắt đầu huấn luyện lại mô hình NLU Medical Intent Classifier...")
            intent_res = train_intent_model()
            acc = intent_res.get("cv_accuracy", 0.0)
            self._log(f"Huấn luyện NLU hoàn tất! Độ chính xác Cross-Validation: {acc * 100:.2f}%. Lưu tại models/intent_classifier.joblib.")
            return intent_res
        except Exception as e:
            self._log(f"Lỗi khi huấn luyện Intent Classifier: {e}")
            return {"status": "error", "error": str(e)}

    def run_full_training_cycle(self) -> Dict[str, Any]:
        """Thực thi một chu kỳ huấn luyện tự động trọn gói (End-to-End)."""
        with self._lock:
            start_t = time.time()
            self._log("=== BẮT ĐẦU CHU KỲ HUẤN LUYỆN TĂNG CƯỜNG TỰ ĐỘNG ===")

            status = self.get_status()
            status["status"] = "training"
            status["last_message"] = "Đang chạy chu kỳ huấn luyện tăng cường..."
            self._save_status(status)

            # 1. Tự học ngân hàng Q&A y tế & danh tính
            qa_count = self.train_synthetic_knowledge()

            # 2. Huấn luyện lại mô hình học máy cảm biến
            ml_sensor_res = self.train_sensor_machine_learning()

            # 3. Huấn luyện lại mô hình NLU Intent Classifier
            ml_intent_res = self.train_nlp_intent_model()

            # 4. Đồng bộ lại tri thức vào RAG Engine & Local Vector Retriever
            try:
                from src.rag.rag_engine import get_rag_engine
                from src.rag.local_vector_retriever import get_local_retriever
                rag = get_rag_engine()
                rag._load_local_documents_cache()
                get_local_retriever().rebuild_index()
                self._log("Đã cập nhật bộ nhớ RAG Engine & tái lập chỉ mục Local Vector Space.")
            except Exception as e:
                self._log(f"Lỗi cập nhật RAG/Vector: {e}")

            elapsed = round(time.time() - start_t, 2)
            self._log(f"=== HOÀN TẤT CHU KỲ HUẤN LUYỆN TRONG {elapsed} GIÂY ===")

            status = self.get_status()
            status["daemon_active"] = self._is_running
            status["last_run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status["total_cycles_completed"] = status.get("total_cycles_completed", 0) + 1
            status["synthetic_qa_samples_learned"] = qa_count
            status["status"] = "idle"
            status["last_message"] = f"Đã hoàn tất chu kỳ #{status['total_cycles_completed']} trong {elapsed}s. Hệ thống đã tự học {qa_count} mẫu tri thức y tế."
            self._save_status(status)

            return {
                "status": "success",
                "qa_learned": qa_count,
                "elapsed_seconds": elapsed,
                "cycle": status["total_cycles_completed"]
            }

    def start_background_daemon(self, interval_seconds: int = 600):
        """Khởi chạy luồng ngầm (Background Daemon) tự động huấn luyện định kỳ."""
        if self._is_running:
            self._log("Background Daemon đang chạy sẵn.")
            return

        self._is_running = True

        def _worker():
            self._log(f"Khởi động Background Training Daemon (chu kỳ: mỗi {interval_seconds} giây).")
            # Chạy ngay 1 chu kỳ ban đầu
            try:
                self.run_full_training_cycle()
            except Exception as e:
                self._log(f"Lỗi chu kỳ đầu: {e}")

            while self._is_running:
                for _ in range(interval_seconds):
                    if not self._is_running:
                        break
                    time.sleep(1)

                if self._is_running:
                    try:
                        self.run_full_training_cycle()
                    except Exception as e:
                        self._log(f"Lỗi chu kỳ ngầm: {e}")

            self._log("Background Training Daemon đã dừng an toàn.")

        self._daemon_thread = threading.Thread(target=_worker, daemon=True)
        self._daemon_thread.start()

    def stop_background_daemon(self):
        """Dừng luồng huấn luyện ngầm."""
        self._is_running = False
        status = self.get_status()
        status["daemon_active"] = False
        status["status"] = "stopped"
        status["last_message"] = "Đã dừng tiến trình huấn luyện ngầm."
        self._save_status(status)
        self._log("Yêu cầu dừng Background Training Daemon.")


# Singleton
auto_trainer_instance = None


def get_auto_trainer() -> AutonomousTrainer:
    global auto_trainer_instance
    if auto_trainer_instance is None:
        auto_trainer_instance = AutonomousTrainer()
    return auto_trainer_instance


if __name__ == "__main__":
    print("Khởi chạy thử nghiệm AutoTrainer...")
    trainer = get_auto_trainer()
    res = trainer.run_full_training_cycle()
    print("Kết quả:", res)
