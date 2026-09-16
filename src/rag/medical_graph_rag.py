# -*- coding: utf-8 -*-
"""
MODULE MEDICAL GRAPHRAG (ĐỒ THỊ TRI THỨC Y TẾ & SUY LUẬN BẮC CẦU ĐA BƯỚC)
Dự án: Hệ thống Chatbot AI Giám sát Người cao tuổi
Kiến trúc: Compound MedRAG & GraphRAG

Chức năng:
1. Xây dựng Đồ thị Tri thức Y tế cục bộ (Zero-dependency Medical Knowledge Graph):
   - Nodes: Bệnh nhân, Bệnh mạn tính, Thuốc, Nhóm dược lý, Dị ứng, Triệu chứng khẩn cấp, Phác đồ xử trí.
   - Edges: ALLERGIC_TO, BELONGS_TO_CLASS, CONTRAINDICATED_FOR, EMERGENCY_SIGN_OF, TREATED_BY, HAS_CONDITION.
2. Suy luận bắc cầu nhiều bước (Multi-hop Reasoning):
   - Ví dụ: Bệnh nhân dị ứng Penicillin. Bác sĩ kê Augmentin.
   - GraphRAG tự động truy vết: Augmentin -> Amoxicillin -> Penicillin -> Dị ứng tuyệt đối -> CẤM TUYỆT ĐỐI!
3. Trích xuất mạng đồ thị con (Subgraph Context) để nạp thẳng vào ngữ cảnh Prompt cho LLM.
4. Hoàn toàn 100% Offline, thực thi < 1ms trên CPU trạm biên.
"""

import os
import sys
import re
import json
from typing import List, Dict, Any, Set, Tuple, Optional

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MedicalKnowledgeGraph:
    """Lõi Đồ thị Tri thức Y tế Lão khoa thuần Python siêu nhẹ."""

    def __init__(self):
        # nodes: dict of node_id -> {type, label, properties}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # edges: list of (source, relation, target, properties)
        self.edges: List[Dict[str, Any]] = []
        # adjacency list: source -> list of (relation, target)
        self.adj: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        # reverse adjacency: target -> list of (relation, source)
        self.rev_adj: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}

        self._build_default_graph()

    def add_node(self, node_id: str, node_type: str, label: str, properties: Optional[Dict[str, Any]] = None):
        """Thêm một thực thể (Node) vào đồ thị."""
        n_id = node_id.strip().lower()
        self.nodes[n_id] = {
            "id": n_id,
            "type": node_type,
            "label": label,
            "properties": properties or {}
        }
        if n_id not in self.adj:
            self.adj[n_id] = []
        if n_id not in self.rev_adj:
            self.rev_adj[n_id] = []

    def add_edge(self, source: str, relation: str, target: str, properties: Optional[Dict[str, Any]] = None):
        """Thêm một mối quan hệ có hướng (Edge) vào đồ thị."""
        s = source.strip().lower()
        t = target.strip().lower()
        rel = relation.strip().upper()
        props = properties or {}

        edge_data = {"source": s, "relation": rel, "target": t, "properties": props}
        self.edges.append(edge_data)

        if s not in self.adj:
            self.adj[s] = []
        self.adj[s].append((rel, t, props))

        if t not in self.rev_adj:
            self.rev_adj[t] = []
        self.rev_adj[t].append((rel, s, props))

    def _build_default_graph(self):
        """Khởi tạo mạng lưới tri thức y tế lâm sàng chuẩn cho cụ Nguyễn Văn An (82 tuổi)."""
        # 1. Bệnh nhân
        self.add_node("patient_an", "PATIENT", "Cụ Nguyễn Văn An", {"age": 82, "room": "102 - Tầng 1"})

        # 2. Bệnh nền mạn tính
        self.add_node("cond_hypertension", "CONDITION", "Tăng huyết áp vô căn độ 2", {"severity": "HIGH"})
        self.add_node("cond_diabetes", "CONDITION", "Đái tháo đường Type 2", {"severity": "HIGH"})
        self.add_node("cond_osteoarthritis", "CONDITION", "Thoái hóa khớp gối", {"severity": "MEDIUM"})
        self.add_node("cond_fall_history", "CONDITION", "Tiền sử té ngã năm 2024", {"risk": "FALL_RISK"})
        self.add_node("cond_stroke", "EMERGENCY_CONDITION", "Đột quỵ não (Tai biến mạch máu não)", {"critical": True})
        self.add_node("cond_cardiac_crisis", "EMERGENCY_CONDITION", "Cơn loạn nhịp tim / Suy tim cấp", {"critical": True})
        self.add_node("cond_hypoglycemia", "EMERGENCY_CONDITION", "Cơn hạ đường huyết cấp tính", {"critical": True})
        self.add_node("cond_choking", "EMERGENCY_CONDITION", "Sặc nghẹn / Hóc dị vật đường thở", {"critical": True})

        # Quan hệ Bệnh nhân -> Bệnh nền
        self.add_edge("patient_an", "HAS_CONDITION", "cond_hypertension")
        self.add_edge("patient_an", "HAS_CONDITION", "cond_diabetes")
        self.add_edge("patient_an", "HAS_CONDITION", "cond_osteoarthritis")
        self.add_edge("patient_an", "HAS_CONDITION", "cond_fall_history")

        # 3. Dược lý & Nhóm thuốc
        self.add_node("class_penicillin", "DRUG_CLASS", "Nhóm Kháng sinh Beta-lactam (Penicillin)", {"allergy_risk": "FATAL_SHOCK"})
        self.add_node("drug_penicillin", "DRUG", "Penicillin", {})
        self.add_node("drug_amoxicillin", "DRUG", "Amoxicillin (Clamoxyl)", {})
        self.add_node("drug_augmentin", "DRUG", "Augmentin (Amoxicillin + Clavulanate)", {})
        self.add_node("drug_ampicillin", "DRUG", "Ampicillin", {})

        self.add_edge("drug_penicillin", "BELONGS_TO_CLASS", "class_penicillin")
        self.add_edge("drug_amoxicillin", "BELONGS_TO_CLASS", "class_penicillin")
        self.add_edge("drug_augmentin", "CONTAINS_INGREDIENT", "drug_amoxicillin")
        self.add_edge("drug_augmentin", "BELONGS_TO_CLASS", "class_penicillin")
        self.add_edge("drug_ampicillin", "BELONGS_TO_CLASS", "class_penicillin")

        # Quan hệ Dị ứng Tuyệt đối
        self.add_edge("patient_an", "ALLERGIC_TO", "class_penicillin", {"reaction": "Sốc phản vệ nguy kịch tử vong"})
        self.add_edge("patient_an", "ALLERGIC_TO", "drug_penicillin", {"severity": "CRITICAL"})
        self.add_edge("patient_an", "ALLERGIC_TO", "drug_amoxicillin", {"severity": "CRITICAL"})
        self.add_edge("patient_an", "ALLERGIC_TO", "drug_augmentin", {"severity": "CRITICAL"})

        # Nhóm NSAID (Kháng viêm không steroid) & Chống chỉ định
        self.add_node("class_nsaid", "DRUG_CLASS", "Nhóm thuốc kháng viêm không steroid (NSAID)", {"danger": "Xuất huyết dạ dày & suy thận"})
        self.add_node("drug_ibuprofen", "DRUG", "Ibuprofen", {})
        self.add_node("drug_diclofenac", "DRUG", "Diclofenac", {})
        self.add_node("drug_meloxicam", "DRUG", "Meloxicam", {})
        self.add_edge("drug_ibuprofen", "BELONGS_TO_CLASS", "class_nsaid")
        self.add_edge("drug_diclofenac", "BELONGS_TO_CLASS", "class_nsaid")
        self.add_edge("drug_meloxicam", "BELONGS_TO_CLASS", "class_nsaid")

        self.add_edge("class_nsaid", "CONTRAINDICATED_FOR", "cond_hypertension", {"reason": "Tăng huyết áp kịch phát & suy thận"})
        self.add_edge("class_nsaid", "CONTRAINDICATED_FOR", "patient_an", {"reason": "Nguy cơ cao xuất huyết tiêu hóa ở người 82 tuổi"})

        # Thuốc giảm đau an toàn: Paracetamol
        self.add_node("drug_paracetamol", "DRUG", "Paracetamol (Panadol / Efferalgan)", {"max_dose": "3000mg/ngày", "safe": True})
        self.add_edge("drug_paracetamol", "FIRST_LINE_TREATMENT", "cond_osteoarthritis", {"instruction": "500mg/lần, cách 4-6h, tối đa 3000mg/ngày"})

        # 4. Dinh dưỡng & Thói quen nguy hiểm
        self.add_node("food_salt", "NUTRIENT", "Muối / Đồ ăn mặn / Dưa cà muối", {"limit": "< 5g/ngày"})
        self.add_node("food_sugar", "NUTRIENT", "Bánh kẹo ngọt / Nước ngọt có gas / Quả ngọt sấy", {"rule": "Kiêng tuyệt đối"})
        self.add_edge("food_salt", "CONTRAINDICATED_FOR", "cond_hypertension", {"reason": "Giữ nước gây cơn tăng huyết áp kịch phát >= 180 mmHg và vỡ mạch não"})
        self.add_edge("food_sugar", "CONTRAINDICATED_FOR", "cond_diabetes", {"reason": "Gây tăng vọt đường huyết và hôn mê thẩm thấu"})

        self.add_node("habit_late_bath", "HABIT", "Tắm đêm sau 19h", {"rule": "CẤM TUYỆT ĐỐI"})
        self.add_edge("habit_late_bath", "TRIGGERS", "cond_stroke", {"reason": "Co thắt mạch máu đột ngột gây đột quỵ não trong nhà tắm"})

        # 5. Triệu chứng lâm sàng & Phác đồ khẩn cấp
        self.add_node("sym_face_droop", "SYMPTOM", "Méo miệng / Lệch nhân trung", {})
        self.add_node("sym_speech_slur", "SYMPTOM", "Nói đớ / Nói ngọng / Không hiểu lời nói", {})
        self.add_node("sym_arm_weakness", "SYMPTOM", "Tê yếu / Liệt 1 bên tay chân", {})
        self.add_node("sym_fall_crash", "SYMPTOM", "Ngã đập sàn nhà / Va đập mạnh", {})
        self.add_node("sym_choking_airway", "SYMPTOM", "Sặc nghẹn / Tím tái khi ăn cháo", {})

        self.add_node("proto_fast", "PROTOCOL", "Phác đồ Cấp cứu Đột quỵ F.A.S.T (Giờ vàng < 4.5h)", {"phone": "115"})
        self.add_node("proto_cpr", "PROTOCOL", "Hồi sinh tim phổi CPR (Ép sâu 5cm, tần số 100-120 l/p)", {"phone": "115"})
        self.add_node("proto_heimlich", "PROTOCOL", "Thủ thuật Heimlich tống dị vật đường thở", {"action": "Giật mạnh hình chữ J"})
        self.add_node("proto_rule_15_15", "PROTOCOL", "Quy tắc 15-15 hạ đường huyết (15g đường, chờ 15 phút)", {})
        self.add_node("proto_virtual_line", "PROTOCOL", "Kỹ thuật vạch kẻ ảo chống đông cứng dáng đi Parkinson", {})

        self.add_edge("sym_face_droop", "EMERGENCY_SIGN_OF", "cond_stroke")
        self.add_edge("sym_speech_slur", "EMERGENCY_SIGN_OF", "cond_stroke")
        self.add_edge("sym_arm_weakness", "EMERGENCY_SIGN_OF", "cond_stroke")
        self.add_edge("cond_stroke", "TREATED_BY", "proto_fast")

        self.add_edge("sym_choking_airway", "EMERGENCY_SIGN_OF", "cond_choking")
        self.add_edge("cond_choking", "TREATED_BY", "proto_heimlich")

    def trace_drug_safety(self, drug_query: str) -> Dict[str, Any]:
        """
        Suy luận bắc cầu (Multi-hop Trace):
        Truy vết xem thuốc có liên hệ với nhóm dị ứng của Cụ An hoặc bệnh nền hay không.
        """
        q = drug_query.strip().lower()
        matched_drug_nodes = []
        for n_id, n_data in self.nodes.items():
            if n_data["type"] in ["DRUG", "DRUG_CLASS"]:
                if n_id in q or q in n_id or any(word in n_data["label"].lower() for word in q.split()):
                    matched_drug_nodes.append(n_id)

        if not matched_drug_nodes:
            # Tìm kiếm mờ theo từ ngữ
            for kw in ["augmentin", "amoxicillin", "penicillin", "ampicillin", "ibuprofen", "paracetamol", "diclofenac"]:
                if kw in q:
                    matched_drug_nodes.append(f"drug_{kw}" if f"drug_{kw}" in self.nodes else f"class_{kw}")

        results = {
            "drug_queried": drug_query,
            "matched_nodes": matched_drug_nodes,
            "is_contraindicated": False,
            "severity": "NORMAL",
            "reasoning_paths": [],
            "clinical_warning": ""
        }

        # Trực tiếp kiểm tra đường đi đến Dị ứng hoặc Chống chỉ định
        for d_id in matched_drug_nodes:
            # 1. Kiểm tra trực tiếp dị ứng của patient_an
            for rel, src, props in self.rev_adj.get(d_id, []):
                if src == "patient_an" and rel == "ALLERGIC_TO":
                    results["is_contraindicated"] = True
                    results["severity"] = "CRITICAL_FATAL"
                    results["reasoning_paths"].append(f"(patient_an) -[ALLERGIC_TO]-> ({d_id})")
                    results["clinical_warning"] = f"⛔ CẤM DÙNG: Bệnh nhân dị ứng trực tiếp với {self.nodes[d_id]['label']}!"

            # 2. Kiểm tra bắc cầu 2 bước: Drug -> Drug Class -> Allergic
            for rel, target, props in self.adj.get(d_id, []):
                if rel in ["BELONGS_TO_CLASS", "CONTAINS_INGREDIENT"]:
                    # Kiểm tra xem lớp này có bị dị ứng không
                    for rev_rel, src, rev_props in self.rev_adj.get(target, []):
                        if src == "patient_an" and rev_rel == "ALLERGIC_TO":
                            results["is_contraindicated"] = True
                            results["severity"] = "CRITICAL_FATAL"
                            path_str = f"({self.nodes[d_id]['label']}) -[{rel}]-> ({self.nodes[target]['label']}) <-[ALLERGIC_TO]- (Cụ Nguyễn Văn An)"
                            results["reasoning_paths"].append(path_str)
                            results["clinical_warning"] = (
                                f"⛔ CẤM TUYỆT ĐỐI! Thuốc '{self.nodes[d_id]['label']}' liên kết với '{self.nodes[target]['label']}' "
                                "mà bệnh nhân có tiền sử DỊ ỨNG NẶNG TUYỆT ĐỐI. Nguy cơ sốc phản vệ tử vong!"
                            )

                    # Kiểm tra xem lớp này có chống chỉ định với bệnh nền không (NSAID -> Hypertension)
                    for c_rel, c_target, c_props in self.adj.get(target, []):
                        if c_rel == "CONTRAINDICATED_FOR":
                            results["is_contraindicated"] = True
                            if results["severity"] != "CRITICAL_FATAL":
                                results["severity"] = "HIGH_RISK"
                            path_str = f"({self.nodes[d_id]['label']}) -[{rel}]-> ({self.nodes[target]['label']}) -[CONTRAINDICATED_FOR]-> ({self.nodes[c_target]['label']})"
                            results["reasoning_paths"].append(path_str)
                            if not results["clinical_warning"]:
                                results["clinical_warning"] = (
                                    f"⚠️ NGUY HIỂM: '{self.nodes[d_id]['label']}' thuộc nhóm '{self.nodes[target]['label']}', "
                                    f"chống chỉ định cho bệnh nhân do {c_props.get('reason', 'nguy cơ biến chứng nặng')}."
                                )

        return results

    def extract_subgraph_for_query(self, query_text: str) -> Dict[str, Any]:
        """
        Nhận diện thực thể trong câu hỏi và trích xuất mạng lưới đồ thị liên quan (Subgraph Extraction).
        """
        q_lower = query_text.lower()
        active_nodes = set()

        # Quét các thực thể xuất hiện trong câu hỏi
        for n_id, n_data in self.nodes.items():
            label = n_data["label"].lower()
            # Bóc tách từ khóa
            clean_label = re.sub(r'\(.*?\)', '', label).strip()
            keywords = [clean_label] + [w for w in clean_label.split() if len(w) > 2]
            if any(kw in q_lower for kw in keywords if len(kw) >= 3):
                active_nodes.add(n_id)

        # Mở rộng đồ thị 1-hop xung quanh các thực thể tìm thấy
        expanded_edges = []
        visited_nodes = set(active_nodes)

        for n in active_nodes:
            # Cạnh đi ra
            for rel, target, props in self.adj.get(n, []):
                visited_nodes.add(target)
                expanded_edges.append({
                    "source": self.nodes.get(n, {}).get("label", n),
                    "relation": rel,
                    "target": self.nodes.get(target, {}).get("label", target),
                    "reason": props.get("reason") or props.get("reaction") or props.get("action") or ""
                })
            # Cạnh đi vào
            for rel, src, props in self.rev_adj.get(n, []):
                visited_nodes.add(src)
                expanded_edges.append({
                    "source": self.nodes.get(src, {}).get("label", src),
                    "relation": rel,
                    "target": self.nodes.get(n, {}).get("label", n),
                    "reason": props.get("reason") or props.get("reaction") or props.get("action") or ""
                })

        # Format context có cấu trúc
        subgraph_facts = []
        for e in expanded_edges:
            desc = f"• {e['source']} --[{e['relation']}]--> {e['target']}"
            if e["reason"]:
                desc += f" (Ghi chú: {e['reason']})"
            subgraph_facts.append(desc)

        return {
            "num_entities": len(visited_nodes),
            "num_relations": len(expanded_edges),
            "entities": [self.nodes[n]["label"] for n in visited_nodes if n in self.nodes],
            "subgraph_text": "\n".join(subgraph_facts) if subgraph_facts else ""
        }


# Singleton
medical_graph_rag_instance = None


def get_medical_graph_rag() -> MedicalKnowledgeGraph:
    global medical_graph_rag_instance
    if medical_graph_rag_instance is None:
        medical_graph_rag_instance = MedicalKnowledgeGraph()
    return medical_graph_rag_instance


if __name__ == "__main__":
    kg = get_medical_graph_rag()
    print("=" * 70)
    print("KIỂM THỬ MODULE MEDICAL GRAPHRAG (KNOWLEDGE GRAPH REASONING)")
    print("=" * 70)

    print("\n1. Thống kê Đồ thị:")
    print(f" - Tổng số Nodes: {len(kg.nodes)}")
    print(f" - Tổng số Edges: {len(kg.edges)}")

    print("\n2. Kiểm thử Suy luận Bắc cầu Đa bước (Multi-hop Trace cho 'Augmentin'):")
    res = kg.trace_drug_safety("Augmentin")
    print(f" - Chống chỉ định: {res['is_contraindicated']}")
    print(f" - Mức độ nghiêm trọng: {res['severity']}")
    print(f" - Đường suy luận: {res['reasoning_paths']}")
    print(f" - Cảnh báo lâm sàng: {res['clinical_warning']}")

    print("\n3. Trích xuất Đồ thị con (Subgraph Extraction) cho câu hỏi: 'cụ bị méo miệng và nói ngọng':")
    subg = kg.extract_subgraph_for_query("cụ bị méo miệng và nói ngọng")
    print(f" - Số thực thể: {subg['num_entities']}")
    print(f" - Tri thức dạng quan hệ:\n{subg['subgraph_text']}")
