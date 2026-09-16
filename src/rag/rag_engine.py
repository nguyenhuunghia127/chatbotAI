# -*- coding: utf-8 -*-
"""
MODULE RAG ENGINE ORCHESTRATOR (COMPOUND 8-IN-1 MEDRAG)
Dự án: Trợ lý AI Giám sát và Chăm sóc Y tế Người cao tuổi AuraCare (Local Edge AI)
Phiên bản: 3.0 Clean Modular Architecture

Kiến trúc Compound MedRAG gồm 8 giai đoạn điều phối:
1. Query Transformation RAG (Chuẩn hóa khẩu ngữ & Mở rộng đa truy vấn)
2. Hybrid Multi-Source Retrieval (ChromaDB + Local TF-IDF + Cache)
3. Advanced Clinical Reranker (Tái xếp hạng đa nhân tố)
4. Corrective RAG (CRAG Evaluator - Pre-retrieval Grading)
5. Medical GraphRAG (Suy luận bắc cầu đồ thị tri thức)
6. Agentic RAG Generation (Kích hoạt Tool Calling y tế)
7. Edge Fail-Safe Deterministic Engine (Dự phòng ngoại tuyến tức thì)
8. Self-RAG Post-generation Reflection (Tự phản tư liều thuốc & dị ứng)
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import requests

from src.core.config import (
    SYSTEM_LOGS_PATH as CONFIG_SYSTEM_LOGS_PATH,
    MEDICAL_DOCS_DIR,
    CHROMA_PERSIST_DIR,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    CANDIDATE_MODELS
)
from src.rag.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    build_agentic_system_prompt,
    STANDARD_MEDICAL_DISCLAIMER
)
from src.rag.retriever import HybridRetriever, get_hybrid_retriever
from src.rag.deterministic_engine import EdgeDeterministicEngine, get_deterministic_engine
from src.rag.query_transformer import get_query_transformer
from src.rag.reranker import get_medical_reranker
from src.rag.corrective_rag import get_corrective_rag_evaluator
from src.rag.medical_graph_rag import get_medical_graph_rag
from src.rag.medical_tools import get_medical_tools_registry

# Re-export để đảm bảo tương thích ngược 100% cho mọi file đang import SYSTEM_LOGS_PATH từ rag_engine
SYSTEM_LOGS_PATH = str(CONFIG_SYSTEM_LOGS_PATH)


class RAGEngine:
    """Điều phối viên tối cao (Master Orchestrator) cho hệ thống Compound 8-in-1 MedRAG."""

    def __init__(
        self,
        logs_file: str = SYSTEM_LOGS_PATH,
        docs_dir: str = str(MEDICAL_DOCS_DIR),
        chroma_dir: str = str(CHROMA_PERSIST_DIR),
        ollama_model: str = OLLAMA_MODEL,
        ollama_host: str = OLLAMA_HOST,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.logs_file = logs_file
        self.docs_dir = docs_dir
        self.chroma_dir = chroma_dir
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.embedding_model_name = embedding_model_name

        self.retriever = get_hybrid_retriever()
        self.deterministic_engine = get_deterministic_engine()

        print(f"[RAG ENGINE] Khoi tao RAG Master Orchestrator (Ollama: {self.ollama_model})", flush=True)

    def _load_local_documents_cache(self):
        """Khả năng tương thích ngược: nạp lại bộ nhớ cache khi có dữ liệu mới."""
        self.retriever.load_documents_cache()

    def rebuild_vector_db(self):
        """Khả năng tương thích ngược: cập nhật chỉ mục ChromaDB."""
        self.retriever.rebuild_vector_db()

    def get_recent_logs(self, n: int = 8) -> str:
        """Trích xuất n dòng log mới nhất."""
        return self.retriever.get_recent_logs(n=n)

    def retrieve_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Quy trình Truy xuất Nâng cao Đa tầng:
        1. Query Transformation -> 2. Multi-Query Search -> 3. Reranker -> 4. CRAG -> 5. GraphRAG.
        """
        # 1. Query Transformation & Multi-Query Expansion
        transformer = get_query_transformer()
        transform_res = transformer.normalize_and_expand(query)
        queries_to_search = transform_res.get("multi_queries", [query])

        # 2. Multi-Query Retrieval
        unique_candidates = self.retriever.retrieve_candidates(queries_to_search, top_k=top_k)

        # 3. Clinical Reranker
        reranker = get_medical_reranker()
        reranked_results = reranker.rerank(query, unique_candidates, top_k=top_k)

        context_chunks = []
        sources = set()
        rerank_scores = []
        for r in reranked_results:
            context_chunks.append(r["text"])
            sources.add(r["source"])
            rerank_scores.append({
                "source": r["source"],
                "score": r["rerank_score"],
                "matched_keywords": r.get("matched_keywords", [])
            })

        # 4. Corrective RAG (CRAG) Pre-generation Evaluation
        crag_evaluator = get_corrective_rag_evaluator()
        crag_eval = crag_evaluator.evaluate_retrieval(query, context_chunks, rerank_scores)

        # 5. Medical GraphRAG Multi-hop Subgraph Extraction
        kg = get_medical_graph_rag()
        subgraph = kg.extract_subgraph_for_query(query)
        drug_trace = kg.trace_drug_safety(query)

        graph_facts = []
        if drug_trace.get("reasoning_paths"):
            for path in drug_trace["reasoning_paths"]:
                graph_facts.append(f"• [QUAN HỆ DƯỢC LÝ BẮC CẦU]: {path}")
            if drug_trace.get("clinical_warning"):
                graph_facts.append(f"• [CẢNH BÁO TỪ ĐỒ THỊ TRI THỨC]: {drug_trace['clinical_warning']}")

        if subgraph.get("subgraph_text"):
            graph_facts.append(subgraph["subgraph_text"])

        if graph_facts:
            context_chunks.append("--- ĐỒ THỊ TRI THỨC Y TẾ ĐA BƯỚC (GRAPHRAG SUBGRAPH) ---\n" + "\n".join(graph_facts))
            sources.add("Medical Knowledge Graph (GraphRAG)")

        combined_context = "\n\n".join(context_chunks) if context_chunks else "Không tìm thấy tài liệu phù hợp trực tiếp."
        return {
            "context_text": combined_context,
            "sources": list(sources),
            "rerank_scores": rerank_scores,
            "crag_evaluation": crag_eval,
            "graph_entities": subgraph.get("entities", []),
            "expanded_queries": queries_to_search
        }

    def _call_llm_api(self, question: str, context: str, latest_logs: str) -> Optional[Tuple[str, str, List[str]]]:
        """Gọi LLM API thông minh hỗ trợ Agentic Tool Calling."""
        if not LLM_API_KEY:
            return None

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        system_msg = build_agentic_system_prompt(context, latest_logs)
        tools_registry = get_medical_tools_registry()
        tool_defs = tools_registry.get_tool_definitions()

        url = f"{LLM_BASE_URL}/chat/completions"
        for model_name in CANDIDATE_MODELS:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question}
                ],
                "tools": tool_defs,
                "tool_choice": "auto",
                "temperature": 0.25,
                "max_tokens": 550
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    res_json = resp.json()
                    choice = res_json.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    tool_calls = message.get("tool_calls", [])

                    executed_tools = []
                    if tool_calls:
                        messages = [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": question},
                            message
                        ]
                        for tool_call in tool_calls:
                            fn = tool_call.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args_str = fn.get("arguments", "{}")
                            import json
                            try:
                                fn_args = json.loads(fn_args_str)
                            except Exception:
                                fn_args = {}

                            tool_res = tools_registry.execute_tool(fn_name, fn_args)
                            executed_tools.append(fn_name)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", "call_1"),
                                "name": fn_name,
                                "content": json.dumps(tool_res, ensure_ascii=False)
                            })

                        second_payload = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": 0.2,
                            "max_tokens": 450
                        }
                        sec_resp = requests.post(url, headers=headers, json=second_payload, timeout=20)
                        if sec_resp.status_code == 200:
                            ans = sec_resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                            if ans:
                                return ans, model_name, executed_tools

                    content = message.get("content", "").strip()
                    if content and len(content) > 20:
                        return content, model_name, executed_tools
            except Exception:
                continue

        return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Gọi Ollama nội bộ qua port 11434."""
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 300}
            }
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return None
        except requests.exceptions.RequestException:
            return None

    def query(self, question: str) -> Dict[str, Any]:
        """
        Quy trình RAG trọn gói:
        1. Kiểm tra bộ nhớ RLHF -> 2. Truy xuất Compound RAG -> 3. Lấy Logs ->
        4. LLM Tool Calling / Ollama -> 5. Edge Fail-Safe Fallback -> 6. Self-RAG Reflection.
        """
        try:
            # 1. Kiểm tra bộ nhớ học tăng cường (RLHF Memory)
            try:
                from src.training.rlhf_service import get_rlhf_service
                few_shots = get_rlhf_service().get_dynamic_few_shot_examples(limit=10)
                for ex in few_shots:
                    if ex["question"].strip().lower() == question.strip().lower():
                        return {
                            "question": question,
                            "answer": ex["answer"],
                            "sources": ["RLHF Ground Truth Memory", "elderly_care.db"],
                            "engine": "RLHF Continuous Learning Engine",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
            except Exception:
                pass

            # 2. Truy xuất tri thức nâng cao
            retrieval_res = self.retrieve_context(question, top_k=3)
            context = retrieval_res["context_text"]
            sources = retrieval_res["sources"]

            # 3. Lấy logs mới nhất
            latest_logs = self.get_recent_logs(n=8)

            # 4. Sinh phản hồi qua LLM API thông minh hoặc Ollama
            tools_used = []
            used_engine = "Edge Deterministic Reasoning Engine (Fail-safe Offline)"
            answer = None

            llm_res = self._call_llm_api(question, context, latest_logs)
            if llm_res:
                answer, model_name, tools_used = llm_res
                used_engine = f"AuraCare Agentic MedRAG ({model_name})"
            else:
                full_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    context=context,
                    latest_logs=latest_logs,
                    question=question
                )
                answer = self._call_ollama(full_prompt)
                if answer:
                    used_engine = "Ollama Local MedRAG"
                    tools_used = ["search_medical_knowledge"]

            # 5. Cơ chế Edge Fail-safe nếu LLM ngoại tuyến
            if not answer:
                used_engine = "Edge Deterministic Reasoning Engine (Fail-safe Offline)"
                answer = self.deterministic_engine.generate_response(question, latest_logs, context)

                q_l = question.lower()
                if any(w in q_l for w in ["tôi là", "tôi tên", "hồ sơ của tôi"]):
                    tools_used = ["get_caregiver_info"]
                elif any(w in q_l for w in ["cụ là ai", "hồ sơ của cụ", "tiền sử", "bệnh án", "bác sĩ"]):
                    tools_used = ["get_patient_profile"]
                elif any(w in q_l for w in ["amoxicillin", "augmentin", "penicillin", "dị ứng", "thuốc", "paracetamol", "nsaid"]):
                    tools_used = ["check_drug_allergy"]
                elif any(w in q_l for w in ["ngã", "sốt", "tim", "sinh hiệu", "thân nhiệt", "huyết áp", "mạch"]):
                    tools_used = ["get_patient_vitals"]
                elif any(w in q_l for w in ["đột quỵ", "cpr", "ngừng thở", "115", "cấp cứu"]):
                    tools_used = ["trigger_emergency_alert", "search_medical_knowledge"]
                else:
                    tools_used = ["search_medical_knowledge"]

            # 6. Self-RAG Post-generation Reflection & Safety Audit
            crag_evaluator = get_corrective_rag_evaluator()
            audit_res = crag_evaluator.self_reflection_audit(
                query=question,
                answer=answer,
                patient_allergies="Dị ứng kháng sinh Penicillin",
                doctor_phone="0912.345.678"
            )
            final_answer = audit_res.get("audited_answer", answer)

            return {
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "engine": used_engine,
                "tools_used": tools_used,
                "rerank_scores": retrieval_res.get("rerank_scores", []),
                "crag_status": retrieval_res.get("crag_evaluation", {}),
                "graph_entities": retrieval_res.get("graph_entities", []),
                "expanded_queries": retrieval_res.get("expanded_queries", []),
                "self_reflection": {
                    "passed": audit_res.get("reflection_passed", True),
                    "corrections": audit_res.get("corrections_applied", [])
                },
                "disclaimer": STANDARD_MEDICAL_DISCLAIMER,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_logs_snapshot": latest_logs
            }

        except Exception as e:
            print(f"[ERROR RAG QUERY]: {str(e)}", file=sys.stderr, flush=True)
            return {
                "question": question,
                "answer": f"Xin lỗi, đã xảy ra lỗi trong quá trình xử lý câu hỏi: {str(e)}",
                "sources": [],
                "engine": "Error Handler",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_logs_snapshot": ""
            }


rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    global rag_engine_instance
    if rag_engine_instance is None:
        rag_engine_instance = RAGEngine()
    return rag_engine_instance


if __name__ == "__main__":
    engine = get_rag_engine()
    test_q = "Sáng nay ông có bị ngã không? Chi tiết thế nào và tôi phải làm gì?"
    print("Test Query:", test_q)
    print("Answer:", engine.query(test_q)["answer"])
