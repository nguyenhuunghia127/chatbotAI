# -*- coding: utf-8 -*-
"""
ENTRYPOINT CHÍNH CỦA MÁY CHỦ FASTAPI BACKEND
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi (Local 100%)

Cách chạy:
    python main_api.py
    hoặc
    uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys

# Đảm bảo đường dẫn gốc được nhận diện chính xác
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.api.main_api import app

if __name__ == "__main__":
    import uvicorn
    print("=" * 80, flush=True)
    print("🚀 [FASTAPI SERVER] Khởi chạy Hệ thống Giám sát Người cao tuổi Local", flush=True)
    print("🌐 Endpoint: http://127.0.0.1:8001", flush=True)
    print("📚 Swagger UI: http://127.0.0.1:8001/docs", flush=True)
    print("=" * 80, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
