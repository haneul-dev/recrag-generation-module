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
import traceback

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
    prov = cfg.get("_provenance", {})
    return {
        "experiment_id": cfg["data"].get("experiment_id", "EXP-GEN-RAW-001"),
        "run_tag": prov.get("run_tag"),
        "snapshot_source": prov.get("snapshot_source"),
        "is_interim": prov.get("is_interim"),
        "run_id": run_id,
        "is_warmup": is_warmup,
        "query_id": query["query_id"],
        "query_text": query["query"],
        "llm_model": cfg["model"]["llm_name"],
        "quantization": (cfg["model"].get("quantization") or "none"),
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


def run_one(cfg, runner, query, filt, run_id, is_warmup, force_error=False) -> dict:
    """단일 (query × run) 실행 -> 로그 1행.

    force_error=True 이면 생성 직전에 예외를 던져 error 경로를 강제 검증한다(테스트용).
    """
    rec = _base_record(cfg, query, filt, run_id, is_warmup)
    timer = ll.LatencyTimer()

    timer.mark("t0_generation_input_received")

    if filt["all_content_missing"]:
        # ── 근거 부족 경로: LLM 호출 없음 (deterministic abstain, 정상 완료) ──
        timer.mark("t1_context_formatting_started")
        timer.mark("t2_context_formatting_finished")
        timer.mark("t3_llm_generation_started")
        timer.mark("t4_llm_generation_finished")
        rec.update(
            {
                "status": "success",
                "row_kind": "deterministic_abstain",
                "llm_invoked": False,
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
            if force_error:
                # 검증용: error 경로(status=error row 기록)를 강제로 발생시킨다.
                raise RuntimeError("forced error for verification (--force-error-on)")
            gen = runner.generate(messages)
            timer.mark("t4_llm_generation_finished")

            parsed = output_parser.parse_generation_output(
                gen["output_text"], filt["used_chunk_ids"]
            )
            timer.mark("t5_postprocessing_finished")

            rec.update(
                {
                    "status": "success",
                    "row_kind": "llm_success",
                    "llm_invoked": True,
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
            tb = traceback.format_exc()
            # AttributeError 처럼 메시지가 비는 경우를 대비해 traceback 마지막 줄까지 확보
            msg = str(e).strip() or repr(e)
            if not msg or msg == f"{type(e).__name__}()":
                last = [ln for ln in tb.strip().splitlines() if ln.strip()]
                msg = last[-1] if last else type(e).__name__
            rec.update(
                {
                    "status": "error",
                    "row_kind": "llm_error",
                    "llm_invoked": True,
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
                    "error_message": msg,
                }
            )
            timer.mark("t5_postprocessing_finished")
            print(f"    [ERROR] {query['query_id']} run={run_id}: {type(e).__name__}: {msg}")
            # 첫 실패는 전체 traceback 을 출력해 원인 진단을 돕는다
            if not getattr(run_one, "_tb_printed", False):
                print("    ----- full traceback (first error) -----")
                print(tb)
                run_one._tb_printed = True

    # 타임스탬프 + 파생 latency
    ts = timer.as_dict()
    rec.update(ts)
    lat = ll.compute_latencies(ts)
    rec.update(lat)
    rec["tokens_per_second"] = ll.compute_tokens_per_second(
        rec["output_token_count"], lat["generation_latency_ms"]
    )
    return rec


def _parse_bool(s):
    """문자열 true/false 를 bool 로. None 이면 None 반환(미지정)."""
    if s is None:
        return None
    return str(s).strip().lower() in ("true", "1", "yes", "y")


def _output_paths(cfg, run_tag: str | None) -> tuple[str, str, str]:
    """run_tag 별 출력 디렉터리를 분리한다.

    run_tag 있으면 : outputs/<run_tag>/{generation_results.jsonl,csv, run_metadata.json}
    run_tag 없으면 : outputs/{generation_results.jsonl,csv, run_metadata.json} (기존 위치)
    → interim/full 산출물이 섞이거나 덮어쓰이지 않는다.
    """
    base_dir = os.path.dirname(cfg["logging"]["jsonl_path"])  # outputs/ (절대경로)
    out_dir = os.path.join(base_dir, run_tag) if run_tag else base_dir
    jsonl_path = os.path.join(out_dir, "generation_results.jsonl")
    csv_path = os.path.join(out_dir, "generation_results.csv")
    meta_path = os.path.join(out_dir, "run_metadata.json")
    return jsonl_path, csv_path, meta_path


def _quantization_effective(q: str) -> str:
    q = (q or "none").lower()
    if q == "4bit":
        return "4bit (nf4, compute fp16)"
    if q == "8bit":
        return "8bit"
    return "none (fp16/bf16, torch_dtype=auto)"


def main():
    parser = argparse.ArgumentParser(description="RECRAG 생성 모듈 Raw Baseline 실행")
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generation_experiment_config.yaml",
    )
    parser.add_argument("--config", default=default_cfg, help="config yaml 경로")
    parser.add_argument("--eval-set", default=None,
                        help="평가셋 경로 override (검증용 jsonl 지정)")
    parser.add_argument("--force-error-on", default=None,
                        help="지정 query_id 의 생성에서 강제 오류 발생(error 경로 검증용)")
    parser.add_argument("--run-tag", default=None,
                        help="출력 파일명 접미사(예: verify) — 본 실행 산출물과 분리 저장")
    parser.add_argument("--validate-only", action="store_true",
                        help="모델 로드 없이 eval set 스키마/라벨 품질만 점검하고 종료")
    parser.add_argument("--experiment-id", default=None,
                        help="experiment_id override (예: interim 실행 구분용)")
    parser.add_argument("--snapshot-source", default=None,
                        help="retrieved_chunks 출처 표기(예: interim_manual / retrieval_topk_v1)")
    parser.add_argument("--is-interim", default=None,
                        help="interim 예비 실행 여부 (true/false). 미지정 시 run_tag 로 추론")
    args = parser.parse_args()

    cfg = config_loader.load_config(args.config)
    if args.eval_set:
        cfg["data"]["eval_set_path"] = os.path.abspath(args.eval_set)
    if args.experiment_id:
        cfg["data"]["experiment_id"] = args.experiment_id

    # ── provenance 결정 (interim/full 오인 방지) ──
    is_interim = _parse_bool(args.is_interim)
    if is_interim is None:  # 미지정 시 run_tag 에 'interim' 포함 여부로 추론
        is_interim = bool(args.run_tag and "interim" in args.run_tag.lower())
    snapshot_source = args.snapshot_source or ("interim_manual" if is_interim else "unspecified")
    cfg["_provenance"] = {
        "run_tag": args.run_tag,
        "snapshot_source": snapshot_source,
        "is_interim": is_interim,
    }

    # ── A안: 검증만 수행 후 종료 (모델 로드 안 함) ──
    if args.validate_only:
        import validate_eval_set as ves
        result = ves.validate_eval_set(cfg["data"]["eval_set_path"])
        ves.print_report(result, cfg["data"]["eval_set_path"])
        sys.exit(1 if result["status"] == "FAIL" else 0)
    print(f"[config] {args.config}")
    print(f"[config] model={cfg['model']['hf_model_id']} "
          f"quantization={cfg['model'].get('quantization') or 'none'} "
          f"context_type={cfg['prompt']['context_type']} "
          f"prompt_type={cfg['prompt']['prompt_type']} "
          f"top_k={cfg['context']['top_k_input']}")

    eval_set = data_loader.load_eval_set(cfg["data"]["eval_set_path"])
    print(f"[data] {len(eval_set)} queries loaded: {cfg['data']['eval_set_path']}")

    # 모델 로드 (지연 import: yaml만 검증할 땐 무거운 로드를 피함)
    from llm_runner import LLMRunner
    import torch

    print("[model] loading ... (Colab GPU)")
    runner = LLMRunner(cfg).load()
    print("[model] loaded.")

    # ── peak VRAM 측정 시작점 리셋 (생성 루프 직전) ──
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()

    warmup_runs = int(cfg["experiment"]["warmup_runs"])
    repeats = int(cfg["experiment"]["repeats"])

    # 출력 경로 (run-tag 별 디렉터리 분리)
    jsonl_path, csv_path, meta_path = _output_paths(cfg, args.run_tag)
    ll.init_jsonl(jsonl_path)
    prov = cfg["_provenance"]
    print(f"[prov] is_interim={prov['is_interim']} snapshot_source={prov['snapshot_source']} "
          f"run_tag={prov['run_tag']} experiment_id={cfg['data'].get('experiment_id')}")
    print(f"[save] incremental JSONL -> {jsonl_path}")
    if args.force_error_on:
        print(f"[verify] force-error-on = {args.force_error_on} (error 경로 검증 모드)")

    all_records = []
    for query in eval_set:
        filt = data_loader.filter_chunks(query["retrieved_chunks"])
        force_error = (args.force_error_on is not None
                       and query["query_id"] == args.force_error_on)
        status = "ABSTAIN(근거부족)" if filt["all_content_missing"] else "generate"
        print(f"\n[{query['query_id']}] {status} | "
              f"used={len(filt['used_chunk_ids'])}/{len(filt['retrieved_chunk_ids'])} chunks "
              f"(content_missing={filt['content_missing_chunk_count']})")

        query_records = []
        # warmup (집계 제외) + 측정 repeats. 각 run 결과를 즉시 JSONL append.
        for w in range(warmup_runs):
            rec = run_one(cfg, runner, query, filt, run_id=f"warmup{w+1}",
                          is_warmup=True, force_error=force_error)
            ll.append_jsonl(rec, jsonl_path)   # 결과 유실 방지: 즉시 기록
            query_records.append(rec)
        for r in range(repeats):
            rec = run_one(cfg, runner, query, filt, run_id=f"r{r+1:02d}",
                          is_warmup=False, force_error=force_error)
            ll.append_jsonl(rec, jsonl_path)
            query_records.append(rec)

        all_records.extend(query_records)
        # CSV는 query 단위로 누적분을 다시 써서 증분 반영
        ll.write_csv(all_records, csv_path)

        # query 별 측정 run latency 중앙값 출력 (LLM 호출 성공 row 만)
        gen_lats = [
            rec["generation_latency_ms"]
            for rec in query_records
            if not rec["is_warmup"]
            and rec.get("row_kind") == "llm_success"
            and rec["generation_latency_ms"] is not None
        ]
        if gen_lats:
            print(f"    generation_latency median = {statistics.median(gen_lats):.1f} ms "
                  f"(n={len(gen_lats)})")
        elif filt["all_content_missing"]:
            print("    (deterministic abstain — LLM latency 집계 제외)")

    # ── peak VRAM 측정 종료 + 집계 ──
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if gpu_available else None
    summary = ll.summarize_generation_latency(all_records)

    from datetime import datetime, timezone
    metadata = {
        "experiment_id": cfg["data"].get("experiment_id"),
        "run_tag": args.run_tag,
        "snapshot_source": cfg["_provenance"]["snapshot_source"],
        "is_interim": cfg["_provenance"]["is_interim"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm_model": cfg["model"]["llm_name"],
        "hf_model_id": cfg["model"]["hf_model_id"],
        "quantization": cfg["model"].get("quantization") or "none",
        "quantization_effective": _quantization_effective(cfg["model"].get("quantization")),
        "decoding_params": config_loader.decoding_params_snapshot(cfg),
        "device": gpu_name,
        "gpu_available": gpu_available,
        "peak_vram_gb": peak_vram_gb,
        "eval_set_path": cfg["data"]["eval_set_path"],
        "n_queries": len(eval_set),
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "force_error_on": args.force_error_on,
        **summary,
    }
    ll.write_json(metadata, meta_path)

    # ── 요약 출력 ──
    c = summary["counts"]
    lat = summary["latency_summary_llm_success_measured"]
    print(f"\n[save] JSONL    -> {jsonl_path}")
    print(f"[save] CSV      -> {csv_path}")
    print(f"[save] metadata -> {meta_path}")
    print(f"[env]  device={gpu_name} quantization={metadata['quantization_effective']} "
          f"peak_vram_gb={peak_vram_gb}")
    print(f"[rows] total={c['total_rows']} warmup={c['warmup_rows']} "
          f"llm_success={c['llm_success']} llm_error={c['llm_error']} "
          f"deterministic_abstain={c['deterministic_abstain']}")
    print(f"[latency] (llm_success, measured) n={lat['n']} "
          f"median={lat['median_ms']}ms p95={lat['p95_ms']}ms "
          f"min={lat['min_ms']}ms max={lat['max_ms']}ms")
    print(f"[format] compliance(measured) = {summary['format_compliance_rate_measured_pct']}%")
    print(f"[done] {len(all_records)} rows "
          f"({len(eval_set)} queries × (warmup {warmup_runs} + repeats {repeats}))")


if __name__ == "__main__":
    main()
