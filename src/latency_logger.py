"""latency_logger.py — t0~t5 latency 계측 + 결과 저장 (M7 Latency Logger).

책임(generation_latency_logging_schema.md):
- monotonic clock 기반 타임스탬프 t0~t5 수집
- §4 계산식으로 파생 latency 산출
- 무결성 검증 (t0≤…≤t5, gap_overhead_ms ≥ 0)
- JSONL 1차 저장 + CSV 파생
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd


def now_ms() -> float:
    """monotonic clock 기준 밀리초."""
    return time.perf_counter() * 1000.0


class LatencyTimer:
    """t0~t5 타임스탬프 수집기. 각 구간 경계에서 mark 를 호출한다."""

    STEPS = [
        "t0_generation_input_received",
        "t1_context_formatting_started",
        "t2_context_formatting_finished",
        "t3_llm_generation_started",
        "t4_llm_generation_finished",
        "t5_postprocessing_finished",
    ]

    def __init__(self):
        self.ts: dict[str, float] = {}

    def mark(self, step: str) -> None:
        if step not in self.STEPS:
            raise ValueError(f"알 수 없는 타임스탬프 단계: {step}")
        self.ts[step] = now_ms()

    def as_dict(self) -> dict:
        return dict(self.ts)


def compute_latencies(ts: dict) -> dict:
    """타임스탬프에서 파생 latency 와 무결성 플래그를 계산한다."""
    t0 = ts.get("t0_generation_input_received")
    t1 = ts.get("t1_context_formatting_started")
    t2 = ts.get("t2_context_formatting_finished")
    t3 = ts.get("t3_llm_generation_started")
    t4 = ts.get("t4_llm_generation_finished")
    t5 = ts.get("t5_postprocessing_finished")

    out = {
        "context_formatting_latency_ms": None,
        "generation_latency_ms": None,
        "postprocessing_latency_ms": None,
        "total_generation_module_latency_ms": None,
        "gap_overhead_ms": None,
        "timestamp_order_ok": None,
    }

    if None in (t0, t1, t2, t3, t4, t5):
        return out

    out["context_formatting_latency_ms"] = t2 - t1
    out["generation_latency_ms"] = t4 - t3
    out["postprocessing_latency_ms"] = t5 - t4
    out["total_generation_module_latency_ms"] = t5 - t0

    seq = [t0, t1, t2, t3, t4, t5]
    out["timestamp_order_ok"] = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))

    out["gap_overhead_ms"] = out["total_generation_module_latency_ms"] - (
        out["context_formatting_latency_ms"]
        + out["generation_latency_ms"]
        + out["postprocessing_latency_ms"]
    )
    return out


def compute_tokens_per_second(output_token_count, generation_latency_ms) -> float | None:
    """초당 출력 토큰. 토큰수/latency 가 없거나 0이면 None (0 나눗셈·None 방지)."""
    if output_token_count is None:
        return None
    if not generation_latency_ms or generation_latency_ms <= 0:
        return None
    return output_token_count / (generation_latency_ms / 1000.0)


# ── 저장 ────────────────────────────────────────────────────
def write_jsonl(records: list[dict], path: str) -> None:
    """전체 records 를 JSONL로 일괄 저장(덮어쓰기)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def init_jsonl(path: str) -> None:
    """증분 저장 시작 전 JSONL 파일을 비운다(이전 실행 결과 누적 방지)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").close()


def append_jsonl(record: dict, path: str) -> None:
    """결과 1행을 JSONL에 즉시 append (중간 오류 시 결과 유실 방지)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# CSV 평탄화 컬럼 (latency_logging_schema §6.3)
CSV_COLUMNS = [
    "experiment_id", "run_id", "is_warmup", "status", "row_kind", "llm_invoked",
    "query_id", "llm_model", "quantization",
    "context_type", "prompt_type", "top_k_input",
    "input_token_count", "output_token_count",
    "context_formatting_latency_ms", "generation_latency_ms",
    "postprocessing_latency_ms", "total_generation_module_latency_ms",
    "tokens_per_second", "groundedness_note", "format_compliance",
    "inline_evidence_set_match", "inline_cited_chunk_ids",
    "evidence_block_chunk_ids", "cited_chunk_ids", "relevant_chunk_ids",
    "content_missing_chunk_count", "all_content_missing",
    "error_type", "error_message",
]

# CSV 에서 배열 -> "C001;C004" 직렬화 대상
_LIST_COLUMNS = {
    "inline_cited_chunk_ids", "evidence_block_chunk_ids",
    "cited_chunk_ids", "relevant_chunk_ids",
}


def write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = []
    for rec in records:
        row = {}
        for col in CSV_COLUMNS:
            val = rec.get(col)
            if col in _LIST_COLUMNS and isinstance(val, list):
                val = ";".join(str(x) for x in val)
            row[col] = val
        rows.append(row)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: dict, path: str) -> None:
    """run metadata 등 단일 JSON 객체 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── 집계 헬퍼 ────────────────────────────────────────────────
def _percentile(values: list[float], q: float) -> float | None:
    """선형 보간 백분위수(q=0~100). values 는 비어있지 않다고 가정."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(xs):
        return xs[lo] + (xs[lo + 1] - xs[lo]) * frac
    return xs[lo]


def summarize_generation_latency(records: list[dict]) -> dict:
    """LLM 호출 성공 + 측정(warmup 제외) row 만으로 latency 통계를 낸다.

    deterministic abstain(LLM 미호출) / error / warmup row 는 latency 집계에서 제외하고,
    개수는 별도로 센다.
    """
    measured = [
        r for r in records
        if not r.get("is_warmup")
        and r.get("row_kind") == "llm_success"
        and r.get("generation_latency_ms") is not None
    ]
    gen_lats = [r["generation_latency_ms"] for r in measured]
    fmt_flags = [bool(r.get("format_compliance")) for r in measured]

    counts = {
        "total_rows": len(records),
        "warmup_rows": sum(1 for r in records if r.get("is_warmup")),
        "llm_success": sum(1 for r in records if r.get("row_kind") == "llm_success"),
        "llm_error": sum(1 for r in records if r.get("row_kind") == "llm_error"),
        "deterministic_abstain": sum(
            1 for r in records if r.get("row_kind") == "deterministic_abstain"
        ),
    }

    if gen_lats:
        import statistics as _st
        latency_summary = {
            "n": len(gen_lats),
            "median_ms": round(_st.median(gen_lats), 1),
            "mean_ms": round(_st.fmean(gen_lats), 1),
            "p95_ms": round(_percentile(gen_lats, 95), 1),
            "min_ms": round(min(gen_lats), 1),
            "max_ms": round(max(gen_lats), 1),
        }
    else:
        latency_summary = {"n": 0, "median_ms": None, "mean_ms": None,
                           "p95_ms": None, "min_ms": None, "max_ms": None}

    fmt_rate = round(sum(fmt_flags) / len(fmt_flags) * 100, 1) if fmt_flags else None

    return {
        "counts": counts,
        "latency_summary_llm_success_measured": latency_summary,
        "format_compliance_rate_measured_pct": fmt_rate,
    }
