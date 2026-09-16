# -*- coding: utf-8 -*-
"""
Package core: Cấu hình và tiện ích hạ tầng trung tâm.
"""

from src.core.config import (
    PROJECT_ROOT,
    DATA_DIR,
    SYSTEM_LOGS_PATH,
    CHROMA_PERSIST_DIR,
    MEDICAL_DOCS_DIR,
    DATA_FILE,
    API_BASE_URL,
    VITALS_THRESHOLDS,
    PATIENT_DEFAULT_INFO,
    PRIMARY_CAREGIVER_DEFAULT
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "SYSTEM_LOGS_PATH",
    "CHROMA_PERSIST_DIR",
    "MEDICAL_DOCS_DIR",
    "DATA_FILE",
    "API_BASE_URL",
    "VITALS_THRESHOLDS",
    "PATIENT_DEFAULT_INFO",
    "PRIMARY_CAREGIVER_DEFAULT"
]
