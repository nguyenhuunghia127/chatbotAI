# -*- coding: utf-8 -*-
"""
MODULE QUẢN LÝ CƠ SỞ DỮ LIỆU CỤC BỘ (LOCAL DATABASE SERVICE - SQLITE)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Tác giả: Kiến trúc sư AI & Kỹ sư Edge Systems

Chức năng:
1. Lưu trữ và quản lý hồ sơ người dùng / người giám hộ (Users / Caregivers).
2. Lưu trữ hồ sơ bệnh án người cao tuổi (Patients Profile: Tiền sử bệnh, dị ứng, bác sĩ phụ trách).
3. Hỗ trợ RAG Engine truy vấn trực tiếp thông tin cá nhân ("Tôi là ai?", "Cụ là ai?").
4. Tự động đồng bộ hồ sơ ra file `data/medical_docs/patient_profile.txt` cho Vector DB.
"""

import os
import sqlite3
import threading
from typing import Dict, Any, Optional, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "elderly_care.db")
PROFILE_DOC_PATH = os.path.join(PROJECT_ROOT, "data", "medical_docs", "patient_profile.txt")


class DatabaseService:
    """Quản lý CSDL SQLite3 an toàn luồng và đồng bộ tri thức RAG."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()
        self._seed_default_data()
        self.sync_profile_to_medical_docs()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Khởi tạo bảng Users và Patients."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Bảng Người dùng / Người giám hộ
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    is_current_active INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Bảng Người cao tuổi / Bệnh nhân được giám sát
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    room TEXT NOT NULL,
                    medical_history TEXT,
                    allergies TEXT,
                    blood_type TEXT,
                    doctor_name TEXT,
                    doctor_phone TEXT,
                    emergency_notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()

    def _seed_default_data(self):
        """Khởi tạo dữ liệu mẫu nếu bảng trống."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Kiểm tra Users
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO users (username, full_name, role, relationship, phone, email, is_current_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    "nghia_nguyen",
                    "Nguyễn Hữu Nghĩa",
                    "Người bảo hộ chính",
                    "Con trai trưởng",
                    "0908.123.456",
                    "nghia.nguyen@email.com",
                    1
                ))

            # Kiểm tra Patients
            cursor.execute("SELECT COUNT(*) FROM patients")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO patients (
                        patient_code, full_name, age, gender, room,
                        medical_history, allergies, blood_type,
                        doctor_name, doctor_phone, emergency_notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "#ELD-8402",
                    "Cụ Nguyễn Văn An",
                    82,
                    "Nam",
                    "Phòng 102 - Tầng 1",
                    "Tăng huyết áp vô căn độ 2, Tiểu đường Type 2, Thoái hóa khớp gối, Tiền sử từng té ngã năm 2024",
                    "Dị ứng kháng sinh Penicillin (Tuyệt đối không dùng)",
                    "O+",
                    "BS. CKI Trần Minh Tuấn (Bệnh viện Lão khoa)",
                    "0912.345.678",
                    "Cụ dễ chóng mặt khi thay đổi tư thế đột ngột từ nằm sang đứng. Cần hỗ trợ khi di chuyển ban đêm."
                ))

            conn.commit()
            conn.close()

    def get_current_user(self) -> Dict[str, Any]:
        """Lấy thông tin người dùng đang hoạt động trong phiên."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE is_current_active = 1 LIMIT 1")
            row = cursor.fetchone()
            if not row:
                cursor.execute("SELECT * FROM users LIMIT 1")
                row = cursor.fetchone()
            conn.close()
            return dict(row) if row else {}

    def set_current_user(self, full_name: str, relationship: str = "Người thân", phone: str = "") -> Dict[str, Any]:
        """Cập nhật hoặc đặt tên người dùng hiện tại."""
        clean_name = full_name.strip()
        if not clean_name:
            return self.get_current_user()

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Bỏ active các user cũ
            cursor.execute("UPDATE users SET is_current_active = 0")

            # Tìm xem user đã tồn tại chưa
            cursor.execute("SELECT id FROM users WHERE full_name LIKE ?", (f"%{clean_name}%",))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("UPDATE users SET is_current_active = 1 WHERE id = ?", (existing["id"],))
            else:
                cursor.execute("""
                    INSERT INTO users (username, full_name, role, relationship, phone, is_current_active)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (
                    clean_name.lower().replace(" ", "_"),
                    clean_name,
                    "Người bảo hộ",
                    relationship,
                    phone or "0908.xxx.xxx"
                ))

            conn.commit()
            conn.close()

        self.sync_profile_to_medical_docs()
        return self.get_current_user()

    def get_patient_profile(self) -> Dict[str, Any]:
        """Lấy thông tin hồ sơ bệnh án của người cao tuổi."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else {}

    def update_patient_profile(self, data: Dict[str, Any]) -> bool:
        """Cập nhật thông tin bệnh án người cao tuổi."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            fields = []
            values = []
            for k in ["full_name", "age", "gender", "room", "medical_history", "allergies", "blood_type", "doctor_name", "doctor_phone", "emergency_notes"]:
                if k in data:
                    fields.append(f"{k} = ?")
                    values.append(data[k])

            if fields:
                values.append(1) # ID = 1
                query = f"UPDATE patients SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
                cursor.execute(query, tuple(values))
                conn.commit()
            conn.close()

        self.sync_profile_to_medical_docs()
        return True

    def sync_profile_to_medical_docs(self):
        """Đồng bộ thông tin bệnh án & người giám hộ vào file text để RAG Vector DB lập chỉ mục."""
        try:
            user = self.get_current_user()
            patient = self.get_patient_profile()

            doc_content = f"""# HỒ SƠ QUẢN LÝ BỆNH NHÂN VÀ NGƯỜI GIÁM HỘ (DATABASE EXPORT)
Ngày cập nhật: {patient.get('updated_at', 'Hiện tại')}

## 1. THÔNG TIN NGƯỜI DÙNG / NGƯỜI GIÁM HỘ (CURRENT USER / CAREGIVER)
- Họ và tên người dùng: {user.get('full_name', 'Nguyễn Hữu Nghĩa')}
- Vai trò: {user.get('role', 'Người bảo hộ chính')}
- Mối quan hệ với cụ: {user.get('relationship', 'Con trai trưởng')}
- Số điện thoại liên hệ khẩn cấp: {user.get('phone', '0908.123.456')}
- Email: {user.get('email', 'nghia.nguyen@email.com')}
- Quyền hạn hệ thống: Theo dõi cảm biến thời gian thực, nhận cảnh báo té ngã / sốt / tim mạch, đàm thoại cùng AI.

## 2. HỒ SƠ NGƯỜI CAO TUỔI ĐƯỢC GIÁM SÁT (PATIENT PROFILE)
- Họ và tên: {patient.get('full_name', 'Cụ Nguyễn Văn An')}
- Mã bệnh nhân: {patient.get('patient_code', '#ELD-8402')}
- Tuổi: {patient.get('age', 82)} tuổi
- Giới tính: {patient.get('gender', 'Nam')}
- Vị trí cư trú: {patient.get('room', 'Phòng 102 - Tầng 1')}
- Nhóm máu: {patient.get('blood_type', 'O+')}
- Tiền sử bệnh án: {patient.get('medical_history', 'Tăng huyết áp vô căn, Tiểu đường Type 2, Thoái hóa khớp, Từng té ngã 2024')}
- Dị ứng thuốc: {patient.get('allergies', 'Dị ứng kháng sinh Penicillin')}
- Bác sĩ điều trị phụ trách: {patient.get('doctor_name', 'BS. CKI Trần Minh Tuấn')} (SĐT: {patient.get('doctor_phone', '0912.345.678')})
- Lưu ý an toàn khẩn cấp: {patient.get('emergency_notes', 'Dễ chóng mặt khi đổi tư thế đột ngột. Cần hỗ trợ di chuyển ban đêm.')}
"""
            os.makedirs(os.path.dirname(PROFILE_DOC_PATH), exist_ok=True)
            with open(PROFILE_DOC_PATH, "w", encoding="utf-8") as f:
                f.write(doc_content)
        except Exception as e:
            print(f"[DB SYNC ERROR]: {e}", flush=True)

    def format_user_summary(self) -> str:
        """Tạo chuỗi tóm tắt thông tin người dùng cho RAG response."""
        u = self.get_current_user()
        p = self.get_patient_profile()
        name = u.get("full_name", "Nguyễn Hữu Nghĩa")
        role = u.get("role", "Người bảo hộ")
        rel = u.get("relationship", "Con trai")
        phone = u.get("phone", "0908.123.456")
        p_name = p.get("full_name", "Cụ Nguyễn Văn An")
        room = p.get("room", "Phòng 102")

        return (
            f"👤 **BẠN LÀ**: **{name}**\n"
            f"• **Vai trò**: {role} ({rel}) của **{p_name}** ({room}).\n"
            f"• **Số điện thoại**: `{phone}`\n"
            f"• **Quyền hạn**: Theo dõi giám sát cảm biến & nhận cảnh báo té ngã, sốt, tim mạch của cụ."
        )

    def format_patient_summary(self) -> str:
        """Tạo chuỗi tóm tắt thông tin người cao tuổi cho RAG response."""
        p = self.get_patient_profile()
        u = self.get_current_user()
        return (
            f"👵👴 **HỒ SƠ {p.get('full_name', 'Cụ Nguyễn Văn An').upper()}** ({p.get('age', 82)}T • {p.get('patient_code', '#ELD-8402')}):\n"
            f"• **Vị trí**: {p.get('room', 'Phòng 102 - Tầng 1')}\n"
            f"• **Tiền sử bệnh**: {p.get('medical_history', 'Tăng huyết áp, Đau khớp, Tiền sử ngã')}\n"
            f"• **Cảnh báo dị ứng**: ⚠️ **{p.get('allergies', 'Không')}**\n"
            f"• **Bác sĩ phụ trách**: {p.get('doctor_name', 'BS. Trần Minh Tuấn')} (SĐT: `{p.get('doctor_phone', '0912.xxx.xxx')}`)\n"
            f"• **Người bảo hộ**: {u.get('full_name', 'Nguyễn Hữu Nghĩa')} ({u.get('phone', '0908.xxx.xxx')})"
        )


# Singleton
db_service_instance = None


def get_db_service() -> DatabaseService:
    global db_service_instance
    if db_service_instance is None:
        db_service_instance = DatabaseService()
    return db_service_instance
