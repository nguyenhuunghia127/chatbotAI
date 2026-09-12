# -*- coding: utf-8 -*-
"""
MODULE BACKEND FASTAPI RESTFUL API (LOCAL 100%)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Cung cấp các endpoint RESTful để Web Dashboard (Streamlit), Mobile App hoặc IoT Edge Node giao tiếp:
1. POST /api/chat          : Gửi câu hỏi người dùng -> RAG Engine -> Trả về câu trả lời có tổng hợp cảm biến
2. GET  /api/logs          : Lấy danh sách nhật ký hệ thống (system_logs.txt) có lọc theo mức độ & giới hạn
3. POST /api/sensor/reading: Tiếp nhận dữ liệu cảm biến mới từ Edge (Camera/Mic/Temp/BLE) -> Phân loại & Tự động ghi log
4. POST /api/sensor/simulate-step: Đọc 1 mẫu ngẫu nhiên từ mock_sensor_data.txt để mô phỏng sự kiện thời gian thực
5. GET  /api/health        : Kiểm tra trạng thái hoạt động của toàn bộ hệ thống
"""

import os
import sys
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Đảm bảo UTF-8 cho console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project root vào sys.path để import an toàn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.rag_engine import get_rag_engine, SYSTEM_LOGS_PATH
from src.sensor_ml.sensor_classifier import SensorClassifier, DATA_FILE
from src.db.db_service import get_db_service

# ==============================================================================
# KHỞI TẠO FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="Elderly Care AI Monitoring System - Local Edge API",
    description="Backend API phục vụ Hệ thống Giám sát & Chatbot AI Người cao tuổi chạy nội bộ 100%",
    version="1.0.0"
)

# Cấu hình CORS mở cho các client cục bộ (Streamlit, Web, App di động)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo singletons cho Sensor Classifier và RAG Engine
sensor_classifier = SensorClassifier()
rag_engine = get_rag_engine()


# ==============================================================================
# ĐỊNH NGHĨA CÁC SCHEMAS (PYDANTIC MODELS)
# ==============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="Câu hỏi hoặc yêu cầu của người dùng bằng tiếng Việt", min_length=1)
    context_window: Optional[int] = Field(8, description="Số lượng dòng log gần nhất cần trích xuất")


class ChatResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    sources: List[str]
    engine: str
    timestamp: str
    latest_logs_snapshot: Optional[str] = None


class SensorReadingRequest(BaseModel):
    aspect_ratio: float = Field(..., description="Tỉ lệ H/W khung xương từ Camera")
    torso_angle: float = Field(..., description="Góc nghiêng thân người (0-90 độ)")
    vertical_velocity: float = Field(..., description="Vận tốc rơi trọng tâm trục Y (m/s)")
    sound_db: float = Field(..., description="Độ lớn âm thanh môi trường (dB)")
    body_temp: float = Field(..., description="Thân nhiệt đo được (°C)")
    heart_rate: int = Field(..., description="Nhịp tim đo từ vòng BLE (BPM)")
    timestamp: Optional[str] = Field(None, description="Mốc thời gian (YYYY-MM-DD HH:MM:SS)")


class SensorReadingResponse(BaseModel):
    status: str = "success"
    predicted_label: str
    confidence: float
    log_level: str
    event_type: str
    description: str
    log_line: str
    timestamp: str


class LogItem(BaseModel):
    raw_log: str
    timestamp: Optional[str] = None
    level: Optional[str] = None
    event_type: Optional[str] = None
    description: Optional[str] = None


class LogsResponse(BaseModel):
    status: str = "success"
    total: int
    count: int
    logs: List[LogItem]


# ==============================================================================
# CÁC ENDPOINT RESTFUL
# ==============================================================================
@app.get("/", tags=["General"])
def root():
    """Trang chủ API và kiểm tra kết nối nhanh."""
    return {
        "system": "Elderly Care AI Local Edge System",
        "status": "online",
        "version": "1.0.0",
        "mode": "100% Offline Local",
        "docs_url": "/docs"
    }


@app.get("/api/health", tags=["General"])
def health_check():
    """Kiểm tra tình trạng hoạt động chi tiết của các module con."""
    logs_exist = os.path.exists(SYSTEM_LOGS_PATH)
    logs_size = os.path.getsize(SYSTEM_LOGS_PATH) if logs_exist else 0

    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modules": {
            "sensor_classifier": "ready",
            "rag_engine": "ready",
            "system_logs_file": {
                "path": SYSTEM_LOGS_PATH,
                "exists": logs_exist,
                "size_bytes": logs_size
            }
        }
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat & RAG"])
def chat_with_ai(req: ChatRequest):
    """
    Endpoint Chatbot RAG:
    Nhận câu hỏi từ người dùng, truy xuất dữ liệu từ nhật ký cảm biến & cẩm nang y tế,
    sau đó phân tích Sensor Fusion và trả lời tự nhiên, có căn cứ.
    """
    try:
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được để trống.")

        # Gọi RAG Engine
        result = rag_engine.query(req.message)

        return ChatResponse(
            status="success",
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            engine=result["engine"],
            timestamp=result["timestamp"],
            latest_logs_snapshot=result.get("latest_logs_snapshot", "")
        )
    except Exception as e:
        print(f"[API ERROR /api/chat]: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý câu hỏi: {str(e)}"
        )


@app.get("/api/logs", response_model=LogsResponse, tags=["Sensor & Logs"])
def get_system_logs(
    limit: int = Query(50, description="Số lượng dòng log tối đa cần lấy", ge=1, le=500),
    level: Optional[str] = Query(None, description="Lọc theo mức độ cảnh báo: CRITICAL, WARNING, INFO")
):
    """
    Lấy danh sách nhật ký hệ thống từ system_logs.txt.
    Hỗ trợ lọc theo log level và sắp xếp mới nhất lên đầu.
    """
    try:
        if not os.path.exists(SYSTEM_LOGS_PATH):
            return LogsResponse(status="success", total=0, count=0, logs=[])

        parsed_logs: List[LogItem] = []

        with open(SYSTEM_LOGS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Đảo ngược để sự kiện mới nhất lên trước
        lines.reverse()

        for line in lines:
            # Parse dòng log: [TIMESTAMP] [LEVEL] [EVENT_TYPE] [METRICS: ...] -> [DESCRIPTION]
            log_time = None
            log_level = None
            event_type = None
            desc = line

            try:
                parts = line.split("] [")
                if len(parts) >= 2:
                    log_time = parts[0].replace("[", "").strip()
                    log_level = parts[1].split("]")[0].strip()

                if " -> " in line:
                    desc = line.split(" -> ")[-1].strip()

                if "[EVENT_TYPE:" in line or ("[" in line and "]" in line):
                    # Tìm event type
                    for possible in ["FALL_DETECTED", "FEVER_DETECTED", "CARDIAC_ALERT", "NORMAL_ACTIVITY", "SYSTEM_INIT"]:
                        if possible in line:
                            event_type = possible
                            break
            except Exception:
                pass

            # Lọc theo level nếu có yêu cầu
            if level and log_level and level.upper() not in log_level.upper():
                continue

            parsed_logs.append(LogItem(
                raw_log=line,
                timestamp=log_time,
                level=log_level,
                event_type=event_type,
                description=desc
            ))

        total = len(parsed_logs)
        limited_logs = parsed_logs[:limit]

        return LogsResponse(
            status="success",
            total=total,
            count=len(limited_logs),
            logs=limited_logs
        )

    except Exception as e:
        print(f"[API ERROR /api/logs]: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Lỗi đọc nhật ký: {str(e)}")


@app.post("/api/sensor/reading", response_model=SensorReadingResponse, tags=["Sensor & Logs"])
def ingest_sensor_reading(reading: SensorReadingRequest):
    """
    Endpoint tiếp nhận dữ liệu cảm biến đa luồng:
    Thực hiện phân loại qua mô hình Scikit-Learn Random Forest và TỰ ĐỘNG GHI LOG vào file system_logs.txt.
    """
    try:
        sample = {
            "aspect_ratio": reading.aspect_ratio,
            "torso_angle": reading.torso_angle,
            "vertical_velocity": reading.vertical_velocity,
            "sound_db": reading.sound_db,
            "body_temp": reading.body_temp,
            "heart_rate": reading.heart_rate
        }

        res = sensor_classifier.process_and_log(sample, timestamp=reading.timestamp)

        # Cập nhật lại cache tri thức của RAG Engine khi có log mới
        rag_engine._load_local_documents_cache()

        return SensorReadingResponse(
            status="success",
            predicted_label=res["predicted_label"],
            confidence=res["confidence"],
            log_level=res["log_level"],
            event_type=res["event_type"],
            description=res["description"],
            log_line=res["log_line"],
            timestamp=res["timestamp"]
        )

    except Exception as e:
        print(f"[API ERROR /api/sensor/reading]: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Lỗi phân loại cảm biến: {str(e)}")


@app.post("/api/sensor/simulate-step", response_model=SensorReadingResponse, tags=["Sensor & Logs"])
def simulate_single_step():
    """
    Đọc 1 dòng ngẫu nhiên từ mock_sensor_data.txt, gán thời gian hiện tại
    và đưa qua mô hình phân loại để kiểm tra hệ thống theo thời gian thực.
    """
    try:
        if not os.path.exists(DATA_FILE):
            raise HTTPException(status_code=404, detail="Không tìm thấy file mock_sensor_data.txt")

        df = pd.read_csv(DATA_FILE, comment="#")
        sample_row = df.sample(n=1).iloc[0]

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sample = {
            "aspect_ratio": float(sample_row["aspect_ratio"]),
            "torso_angle": float(sample_row["torso_angle"]),
            "vertical_velocity": float(sample_row["vertical_velocity"]),
            "sound_db": float(sample_row["sound_db"]),
            "body_temp": float(sample_row["body_temp"]),
            "heart_rate": int(sample_row["heart_rate"])
        }

        res = sensor_classifier.process_and_log(sample, timestamp=now_str)
        rag_engine._load_local_documents_cache()

        return SensorReadingResponse(
            status="success",
            predicted_label=res["predicted_label"],
            confidence=res["confidence"],
            log_level=res["log_level"],
            event_type=res["event_type"],
            description=res["description"],
            log_line=res["log_line"],
            timestamp=res["timestamp"]
        )

    except Exception as e:
        print(f"[API ERROR /api/sensor/simulate-step]: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Lỗi mô phỏng sự kiện: {str(e)}")


class VoiceTranscribeResponse(BaseModel):
    status: str = "success"
    transcribed_text: str
    language: str = "vi"


@app.post("/api/voice/transcribe", response_model=VoiceTranscribeResponse, tags=["Voice Input (STT)"])
async def transcribe_voice_audio(file: UploadFile = File(...)):
    """
    Endpoint nhận diện giọng nói Tiếng Việt (Speech-to-Text):
    Nhận file âm thanh (.wav, .mp3, .m4a) từ Microphone và chuyển thành văn bản Tiếng Việt bằng Faster-Whisper.
    """
    try:
        from src.voice.voice_service import get_voice_service
        vs = get_voice_service()
        contents = await file.read()
        text = vs.transcribe_audio_bytes(contents, language="vi")
        return VoiceTranscribeResponse(status="success", transcribed_text=text, language="vi")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện giọng nói Tiếng Việt: {str(e)}")


class UserProfileUpdateRequest(BaseModel):
    full_name: str
    relationship: Optional[str] = "Người bảo hộ"
    phone: Optional[str] = "0908.xxx.xxx"


@app.get("/api/user/profile", tags=["User & Profile"])
def get_user_profile():
    """Lấy thông tin người giám hộ hiện tại từ CSDL SQLite."""
    return {"status": "success", "user": get_db_service().get_current_user()}


@app.post("/api/user/profile", tags=["User & Profile"])
def update_user_profile(req: UserProfileUpdateRequest):
    """Cập nhật thông tin danh tính người dùng vào CSDL SQLite."""
    u = get_db_service().set_current_user(req.full_name, relationship=req.relationship, phone=req.phone)
    rag_engine._load_local_documents_cache()
    return {"status": "success", "user": u}


@app.get("/api/patient/profile", tags=["User & Profile"])
def get_patient_profile():
    """Lấy thông tin hồ sơ bệnh án người cao tuổi từ CSDL SQLite."""
    return {"status": "success", "patient": get_db_service().get_patient_profile()}


class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    rating: int = Field(..., description="+1 (Hài lòng) hoặc -1 (Chưa hài lòng)")
    corrected_response: Optional[str] = ""
    feedback_text: Optional[str] = ""


@app.post("/api/feedback", tags=["RLHF & Continuous Learning"])
def submit_feedback(req: FeedbackRequest):
    """Tiếp nhận phản hồi người dùng (+1 / -1 / câu trả lời chuẩn) phục vụ huấn luyện tăng cường (RLHF)."""
    from src.training.rlhf_service import get_rlhf_service
    res = get_rlhf_service().record_feedback(
        prompt=req.prompt,
        response=req.response,
        rating=req.rating,
        corrected_response=req.corrected_response,
        feedback_text=req.feedback_text,
        source="api"
    )
    return res


@app.get("/api/feedback/stats", tags=["RLHF & Continuous Learning"])
def get_feedback_stats():
    """Lấy thống kê phản hồi và tỉ lệ hài lòng của người dùng đối với Chatbot AI."""
    from src.training.rlhf_service import get_rlhf_service
    return {"status": "success", "stats": get_rlhf_service().get_feedback_stats()}


@app.on_event("startup")
def startup_event():
    """Khởi động Background Daemon tự động huấn luyện ngầm định kỳ khi máy chủ API chạy."""
    try:
        from src.training.auto_trainer import get_auto_trainer
        trainer = get_auto_trainer()
        trainer.start_background_daemon(interval_seconds=600)
        print("🤖 [AUTONOMOUS TRAINER] Tiến trình huấn luyện ngầm tự động đã được kích hoạt (chu kỳ 10 phút/lần).", flush=True)
    except Exception as e:
        print(f"⚠️ [AUTONOMOUS TRAINER] Lỗi khởi động background daemon: {e}", flush=True)


@app.on_event("shutdown")
def shutdown_event():
    """Dừng Background Daemon an toàn khi máy chủ tắt."""
    try:
        from src.training.auto_trainer import get_auto_trainer
        get_auto_trainer().stop_background_daemon()
    except Exception:
        pass


@app.post("/api/training/run-auto", tags=["Autonomous Auto-Trainer"])
def run_auto_training_cycle():
    """Kích hoạt 1 chu kỳ huấn luyện ngầm tự động ngay lập tức (tự học tri thức Q&A + retrain ML cảm biến)."""
    from src.training.auto_trainer import get_auto_trainer
    trainer = get_auto_trainer()
    threading.Thread(target=trainer.run_full_training_cycle, daemon=True).start()
    return {"status": "started", "message": "Đã kích hoạt chu kỳ huấn luyện ngầm tự động trong tiến trình nền!"}


@app.post("/api/training/daemon/start", tags=["Autonomous Auto-Trainer"])
def start_auto_training_daemon(interval_seconds: int = 600):
    """Bật tiến trình daemon chạy ngầm tự động huấn luyện định kỳ (mặc định: 600 giây)."""
    from src.training.auto_trainer import get_auto_trainer
    trainer = get_auto_trainer()
    trainer.start_background_daemon(interval_seconds=interval_seconds)
    return {"status": "success", "message": f"Background Training Daemon đã chạy (chu kỳ {interval_seconds}s)!"}


@app.post("/api/training/daemon/stop", tags=["Autonomous Auto-Trainer"])
def stop_auto_training_daemon():
    """Dừng tiến trình daemon chạy ngầm."""
    from src.training.auto_trainer import get_auto_trainer
    trainer = get_auto_trainer()
    trainer.stop_background_daemon()
    return {"status": "success", "message": "Background Training Daemon đã dừng!"}


@app.get("/api/training/status", tags=["Autonomous Auto-Trainer"])
def get_auto_training_status():
    """Xem trạng thái và số chu kỳ huấn luyện ngầm tự động đã hoàn tất."""
    from src.training.auto_trainer import get_auto_trainer
    return {"status": "success", "training_status": get_auto_trainer().get_status()}


class LLMGenerateDataRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="API Key (Mặc định dùng key đã cung cấp)")
    base_url: Optional[str] = Field("https://api.openai.com/v1", description="Đường dẫn Base URL của API")
    model: Optional[str] = Field("gpt-4o-mini", description="Tên mô hình LLM")
    topic: Optional[str] = Field("all", description="Chủ đề cần sinh dữ liệu (hoặc 'all')")
    count_per_topic: Optional[int] = Field(3, description="Số lượng mẫu trên mỗi chủ đề")


@app.post("/api/training/generate-llm-data", tags=["Autonomous Auto-Trainer"])
def generate_llm_training_data(req: LLMGenerateDataRequest):
    """
    Gọi LLM API bên ngoài để tự động sinh dữ liệu đối thoại chuẩn y tế (Prompt - Response - DPO)
    và nạp trực tiếp vào bộ nhớ RLHF & Few-Shot của chatbot.
    """
    try:
        from src.training.llm_synthetic_generator import LLMSyntheticGenerator
        from src.training.llm_synthetic_generator import DEFAULT_API_KEY

        active_key = req.api_key or DEFAULT_API_KEY
        gen = LLMSyntheticGenerator(api_key=active_key, base_url=req.base_url, model=req.model)

        if req.topic == "all":
            res = gen.generate_all_topics(samples_per_topic=req.count_per_topic)
        else:
            samples = gen.generate_qa_pairs(topic_key=req.topic, num_samples=req.count_per_topic)
            res = gen.integrate_into_knowledge_base(samples)
            res["total_generated"] = len(samples)

        # Cập nhật lại cache tri thức của RAG Engine ngay sau khi nạp mẫu mới
        rag_engine._load_local_documents_cache()

        return {
            "status": "success",
            "message": f"Đã sinh thành công {res.get('total_generated', 0)} mẫu dữ liệu huấn luyện!",
            "details": res
        }
    except Exception as e:
        print(f"[API ERROR /api/training/generate-llm-data]: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Lỗi tạo sinh dữ liệu bằng LLM: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 80, flush=True)
    print("🚀 [FASTAPI STARTUP] Khởi chạy máy chủ API tại http://127.0.0.1:8001", flush=True)
    print("📖 [SWAGGER DOCS] Xem tài liệu API tương tác tại http://127.0.0.1:8001/docs", flush=True)
    print("=" * 80, flush=True)
    uvicorn.run("src.api.main_api:app", host="127.0.0.1", port=8001, reload=False)
