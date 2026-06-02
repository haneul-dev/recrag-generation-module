"""run_generation.py — Raw Context Baseline 실행 엔트리포인트.

흐름:
  config 로드 -> 평가셋 로드 -> 모델 로드
  -> 각 query 마다 (warmup_runs + repeats) 회 실행
     [t0] 입력 수신 -> [t1~t2] 컨텍스트 포맷팅 -> [t3~t4] LLM 생성 -> [t5] 파싱
  -> JSONL + CSV 저장 -> 요약 출력

content 필터링 결과 모든 chunk에 content가 없으면(all_content_missing)
LLM 호출 없이 '근거 부족'으로 처리한다.

실행:
  python src/run_generation.py --config generation_experiment_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

# src/ 를 import 경로에 추가 (Colab/로컬 어디서 실행해도 동작)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_loader
import data_loader
import prompt_builder
import output_parser
import latency_logger as ll

# 근거 부족(all_content_missing) 시 표준 응답 (rag_prompt_template §7.1)
ABSTAIN_ANSWER = "제공된 근거 문서에서 해당 질의에 답할 수 있는 내용을 찾지 못했습니다."


def _prompt_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _base_record(cfg, query, filt, run_id, is_warmup) -> dict:
    """식별/입력 공통 컬럼."""
    return {
        "experiment_id": cfg["data"].get("experiment_id", "EXP-GEN-RAW-001"),
        "run_id": run_id,
        "is_warmup": is_warmup,
        "query_id": query["query_id"],
        "query_text": query["query"],
        "llm_model": cfg["model"]["llm_name"],
        "context_type": cfg["prompt"]["context_type"],
        "prompt_type": cfg["prompt"]["prompt_type"],
        "top_k_input": cfg["context"]["top_k_input"],
        "decoding_params": config_loader.decoding_params_snapshot(cfg),
        "retrieved_chunk_ids": filt["retrieved_chunk_ids"],
        "used_chunk_ids": filt["used_chunk_ids"],
        "content_missing_chunk_count": filt["content_missing_chunk_count"],
        "all_content_missing": filt["all_content_missing"],
        "relevant_chunk_ids": query.get("relevant_chunk_ids", []),
    }


def run_one(cfg, runner, query, filt, run_id, is_warmup) -> dict:
    """단일 (query × run) 실행 -> 로그 1행."""
    rec = _base_record(cfg, query, filt, run_id, is_warmup)
    timer = ll.LatencyTimer()

    timer.mark("t0_generation_input_received")

    if filt["all_content_missing"]:
        # ── 근거 부족 경로: LLM 호출 없음 (정상 완료된 기권) ──
        timer.mark("t1_context_formatting_started")
        timer.mark("t2_context_formatting_finished")
        timer.mark("t3_llm_generation_started")
        timer.mark("t4_llm_generation_finished")
        rec.update(
            {
                "status": "success",
                "final_prompt_hash": None,
                "input_token_count": 0,
                "output_token_count": 0,
                "generated_answer": ABSTAIN_ANSWER,
                "inline_cited_chunk_ids": [],
                "evidence_block_chunk_ids": [],
                "cited_chunk_ids": [],
                "groundedness_note": "근거 부족",
                "inline_evidence_set_match": True,
                "format_compliance": True,
                "decoding_note": None,
                "error_type": "all_content_missing",
                "error_message": None,
            }
        )
        timer.mark("t5_postprocessing_finished")
    else:
        # ── 정상 생성 경로 ──
        timer.mark("t1_context_formatting_started")
        messages = prompt_builder.build_messages(query["query"], filt["used_chunks"])
        timer.mark("t2_context_formatting_finished")

        timer.mark("t3_llm_generation_started")
        try:
            gen = runner.generate(messages)
            timer.mark("t4_llm_generation_finished")

            parsed = output_parser.parse_generation_output(
                gen["output_text"], filt["used_chunk_ids"]
            )
            timer.mark("t5_postprocessing_finished")

            rec.update(
                {
                    "status": "success",
                    "final_prompt_hash": _prompt_hash(messages),
                    "final_prompt_text": (
                        json.dumps(messages, ensure_ascii=False)
                        if cfg["logging"].get("save_final_prompt_text")
                        else None
                    ),
                    "input_token_count": gen["input_token_count"],
                    "output_token_count": gen["output_token_count"],
                    "generated_answer": parsed["answer"],
                    "inline_cited_chunk_ids": parsed["inline_cited_chunk_ids"],
                    "evidence_block_chunk_ids": parsed["evidence_block_chunk_ids"],
                    "cited_chunk_ids": parsed["cited_chunk_ids"],
                    "groundedness_note": parsed["groundedness_note"],
                    "inline_evidence_set_match": parsed["inline_evidence_set_match"],
                    "format_compliance": parsed["format_compliance"],
                    "decoding_note": gen["decoding_note"],
                    "error_type": parsed["error_type"],
                    "error_message": None,
                }
            )
        except Exception as e:
            # GPU OOM 등 생성 중 오류 — 해당 query 결과를 error record 로 남기고 계속 진행
            timer.mark("t4_llm_generation_finished")
            rec.update(
                {
                    "status": "error",
                    "final_prompt_hash": _prompt_hash(messages),
                    "input_token_count": None,
                    "output_token_count": None,
                    "generated_answer": None,
                    "inline_cited_chunk_ids": [],
                    "evidence_block_chunk_ids": [],
                    "cited_chunk_ids": [],
                    "groundedness_note": "generation failed before completion",
                    "inline_evidence_set_match": False,
                    "format_compliance": False,
                    "decoding_note": None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            timer.mark("t5_postprocessing_finished")
            print(f"    [ERROR] {query['query_id']} run={run_id}: "
                  f"{type(e).__name__}: {e}")

    # 타임스탬프 + 파생 latency
    ts = timer.as_dict()
    rec.update(ts)
    lat = ll.compute_latencies(ts)
    rec.update(lat)
    rec["tokens_per_second"] = ll.compute_tokens_per_second(
        rec["output_token_count"], lat["generation_latency_ms"]
    )
    return rec


def main():
    parser = argparse.ArgumentParser(description="RECRAG 생성 모듈 Raw Baseline 실행")
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generation_experiment_config.yaml",
    )
    parser.add_argument("--config", default=default_cfg, help="config yaml 경로")
    args = parser.parse_args()

    cfg = config_loader.load_config(args.config)
    print(f"[config] {args.config}")
    print(f"[config] model={cfg['model']['hf_model_id']} "
          f"context_type={cfg['prompt']['context_type']} "
          f"prompt_type={cfg['prompt']['prompt_type']} "
          f"top_k={cfg['context']['top_k_input']}")

    eval_set = data_loader.load_eval_set(cfg["data"]["eval_set_path"])
    print(f"[data] {len(eval_set)} queries loaded: {cfg['data']['eval_set_path']}")

    # 모델 로드 (지연 import: yaml만 검증할 땐 무거운 로드를 피함)
    from llm_runner import LLMRunner

    print("[model] loading ... (Colab GPU)")
    runner = LLMRunner(cfg).load()
    print("[model] loaded.")

    warmup_runs = int(cfg["experiment"]["warmup_runs"])
    repeats = int(cfg["experiment"]["repeats"])

    # 증분 저장 준비: JSONL은 비우고 시작, 이후 query 단위로 즉시 append
    jsonl_path = cfg["logging"]["jsonl_path"]
    csv_path = cfg["logging"]["csv_path"]
    ll.init_jsonl(jsonl_path)
    print(f"[save] incremental JSONL -> {jsonl_path}")

    all_records = []
    error_count = 0
    for query in eval_set:
        filt = data_loader.filter_chunks(query["retrieved_chunks"])
        status = "ABSTAIN(근거부족)" if filt["all_content_missing"] else "generate"
        print(f"\n[{query['query_id']}] {status} | "
              f"used={len(filt['used_chunk_ids'])}/{len(filt['retrieved_chunk_ids'])} chunks "
              f"(content_missing={filt['content_missing_chunk_count']})")

        query_records = []
        # warmup (집계 제외) + 측정 repeats. 각 run 결과를 즉시 JSONL append.
        for w in range(warmup_runs):
            rec = run_one(cfg, runner, query, filt, run_id=f"warmup{w+1}", is_warmup=True)
            ll.append_jsonl(rec, jsonl_path)   # 결과 유실 방지: 즉시 기록
            query_records.append(rec)
        for r in range(repeats):
            rec = run_one(cfg, runner, query, filt, run_id=f"r{r+1:02d}", is_warmup=False)
            ll.append_jsonl(rec, jsonl_path)
            query_records.append(rec)

        all_records.extend(query_records)
        error_count += sum(1 for rec in query_records if rec.get("status") == "error")

        # CSV는 query 단위로 누적분을 다시 써서 증분 반영
        ll.write_csv(all_records, csv_path)

        # query 별 측정 run 의 생성 latency 중앙값 출력 (성공 run 만)
        gen_lats = [
            rec["generation_latency_ms"]
            for rec in query_records
            if not rec["is_warmup"]
            and rec.get("status") == "success"
            and rec["generation_latency_ms"] is not None
        ]
        if gen_lats:
            print(f"    generation_latency median = {statistics.median(gen_lats):.1f} ms "
                  f"(n={len(gen_lats)})")

    print(f"\n[save] JSONL -> {jsonl_path}")
    print(f"[save] CSV   -> {csv_path}")
    print(f"[done] {len(all_records)} rows "
          f"({len(eval_set)} queries × (warmup {warmup_runs} + repeats {repeats})) | "
          f"error rows = {error_count}")


if __name__ == "__main__":
    main()
