"""evaluate.py — held-out test 정량 평가 (base vs 파인튜닝).

지표
----
답가능(answerable):
  - answer_token_f1, answer_char_f1 : 생성 답변 vs ground_truth (표면 유사도)
  - citation_precision/recall/f1     : cited_chunk_ids vs relevant_chunk_ids
  - citation_exact                   : 집합 완전일치 비율
  - format_compliance                : 출력 형식 준수율
  - latency p50/p95 (ms)
답불가(abstain; relevant 없음 or query_id=X*):
  - abstain_ok      : '근거 부족' + 인용 0
  - hallucination   : 근거 없는데 인용/단정
  - format_compliance

메모리 안전: 모델은 1개만 로드한다. base/파인튜닝은 각각 별도 프로세스로 실행하고,
--compare 로 두 summary.json 을 표로 합친다.

실행
----
  python finetune/evaluate.py --config finetune/finetune_config.yaml --tag base
  python finetune/evaluate.py --config finetune/finetune_config.yaml --tag finetuned \
      --adapter finetune/outputs/qwen2.5-7b-recrag-qlora
  python finetune/evaluate.py --compare finetune/outputs/eval/base.summary.json \
      finetune/outputs/eval/finetuned.summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))


# ─────────────────────────────────────────────────────────────
# 지표 헬퍼 (의존성 없이)
# ─────────────────────────────────────────────────────────────
def _word_tokens(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()


def _char_tokens(s: str) -> list[str]:
    return list(re.sub(r"\s+", "", re.sub(r"[^\w\s]", " ", (s or "").lower())))


def _f1(pred: list[str], gold: list[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pred), n / len(gold)
    return 2 * p * r / (p + r)


def _set_prf(cited: set[str], relevant: set[str]) -> tuple[float, float, float]:
    if not cited and not relevant:
        return 1.0, 1.0, 1.0
    inter = len(cited & relevant)
    p = inter / len(cited) if cited else 0.0
    r = inter / len(relevant) if relevant else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _is_abstain_row(row: dict) -> bool:
    return (not row.get("relevant_chunk_ids")) or str(row.get("query_id", "")).upper().startswith("X")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ─────────────────────────────────────────────────────────────
# 평가 실행
# ─────────────────────────────────────────────────────────────
def run_eval(cfg: dict, repo_root: str, test_path: str, adapter: str | None,
             tag: str, warmup: int, results_dir: str) -> dict:
    from data_loader import load_eval_set, filter_chunks
    from prompt_builder import build_messages
    from output_parser import parse_generation_output
    from llm_runner import LLMRunner

    mcfg = dict(cfg["model"])
    mcfg["quantization"] = "4bit"
    if adapter:
        mcfg["adapter_path"] = adapter
    runner = LLMRunner({"model": mcfg}).load()

    rows = load_eval_set(test_path)

    # warmup (latency 집계 제외)
    for _ in range(max(0, warmup)):
        if rows:
            f = filter_chunks(rows[0]["retrieved_chunks"])
            runner.generate(build_messages(rows[0]["query"], f["used_chunks"]))

    per_example, ans_metrics, abs_metrics, latencies = [], [], [], []

    for row in rows:
        f = filter_chunks(row["retrieved_chunks"])
        used_ids = f["used_chunk_ids"]
        msgs = build_messages(row["query"], f["used_chunks"])

        t0 = time.monotonic()
        gen = runner.generate(msgs)
        lat_ms = (time.monotonic() - t0) * 1000.0
        latencies.append(lat_ms)

        parsed = parse_generation_output(gen["output_text"], used_ids)
        is_abs = _is_abstain_row(row)
        rec = {
            "query_id": row.get("query_id"), "group": "abstain" if is_abs else "answerable",
            "format_compliance": parsed["format_compliance"],
            "groundedness_note": parsed["groundedness_note"],
            "cited_chunk_ids": parsed["cited_chunk_ids"],
            "latency_ms": round(lat_ms, 1),
            "output_token_count": gen["output_token_count"],
        }

        if is_abs:
            abstain_ok = (parsed["groundedness_note"] == "근거 부족"
                          and len(parsed["cited_chunk_ids"]) == 0)
            hallucination = len(parsed["cited_chunk_ids"]) > 0
            rec.update({"abstain_ok": abstain_ok, "hallucination": hallucination})
            abs_metrics.append(rec)
        else:
            gt = row.get("ground_truth_answer", "")
            tok_f1 = _f1(_word_tokens(parsed["answer"]), _word_tokens(gt))
            chr_f1 = _f1(_char_tokens(parsed["answer"]), _char_tokens(gt))
            cited = set(parsed["cited_chunk_ids"])
            relevant = set(row.get("relevant_chunk_ids", []))
            cp, crc, cf = _set_prf(cited, relevant)
            rec.update({
                "answer_token_f1": round(tok_f1, 4), "answer_char_f1": round(chr_f1, 4),
                "citation_precision": round(cp, 4), "citation_recall": round(crc, 4),
                "citation_f1": round(cf, 4), "citation_exact": cited == relevant,
            })
            ans_metrics.append(rec)
        per_example.append(rec)

    # ── 집계 ──
    n_ans, n_abs = len(ans_metrics), len(abs_metrics)
    summary = {
        "tag": tag, "adapter": adapter, "test_path": test_path,
        "n_total": len(rows), "n_answerable": n_ans, "n_abstain": n_abs,
        "latency_ms": {"p50": round(_pct(latencies, 0.5), 1),
                       "p95": round(_pct(latencies, 0.95), 1),
                       "mean": round(_mean(latencies), 1)},
        "answerable": {
            "answer_token_f1": round(_mean([m["answer_token_f1"] for m in ans_metrics]), 4),
            "answer_char_f1": round(_mean([m["answer_char_f1"] for m in ans_metrics]), 4),
            "citation_precision": round(_mean([m["citation_precision"] for m in ans_metrics]), 4),
            "citation_recall": round(_mean([m["citation_recall"] for m in ans_metrics]), 4),
            "citation_f1": round(_mean([m["citation_f1"] for m in ans_metrics]), 4),
            "citation_exact_rate": round(_mean([1.0 if m["citation_exact"] else 0.0 for m in ans_metrics]), 4),
            "format_compliance": round(_mean([1.0 if m["format_compliance"] else 0.0 for m in ans_metrics]), 4),
        } if n_ans else {},
        "abstain": {
            "abstain_ok_rate": round(_mean([1.0 if m["abstain_ok"] else 0.0 for m in abs_metrics]), 4),
            "hallucination_rate": round(_mean([1.0 if m["hallucination"] else 0.0 for m in abs_metrics]), 4),
            "format_compliance": round(_mean([1.0 if m["format_compliance"] else 0.0 for m in abs_metrics]), 4),
        } if n_abs else {},
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{tag}.perexample.jsonl"), "w", encoding="utf-8") as fp:
        for r in per_example:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(results_dir, f"{tag}.summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {results_dir}/{tag}.summary.json")
    return summary


# ─────────────────────────────────────────────────────────────
# 비교표
# ─────────────────────────────────────────────────────────────
def _row(label, base, ft, fmt="{:.4f}"):
    b = fmt.format(base) if isinstance(base, (int, float)) else str(base)
    f = fmt.format(ft) if isinstance(ft, (int, float)) else str(ft)
    delta = ""
    if isinstance(base, (int, float)) and isinstance(ft, (int, float)):
        d = ft - base
        delta = ("+" if d >= 0 else "") + fmt.format(d)
    return f"| {label} | {b} | {f} | {delta} |"


def compare(base_json: str, ft_json: str) -> None:
    b = json.load(open(base_json, encoding="utf-8"))
    f = json.load(open(ft_json, encoding="utf-8"))
    print(f"\n# base vs 파인튜닝 비교 (held-out test, n={b['n_total']})\n")
    print("## 답가능")
    print("| 지표 | base | 파인튜닝 | Δ |")
    print("|---|---|---|---|")
    for k in ["answer_token_f1", "answer_char_f1", "citation_precision", "citation_recall",
              "citation_f1", "citation_exact_rate", "format_compliance"]:
        print(_row(k, b["answerable"].get(k, 0), f["answerable"].get(k, 0)))
    print("\n## 답불가(abstain)")
    print("| 지표 | base | 파인튜닝 | Δ |")
    print("|---|---|---|---|")
    for k in ["abstain_ok_rate", "hallucination_rate", "format_compliance"]:
        print(_row(k, b["abstain"].get(k, 0), f["abstain"].get(k, 0)))
    print("\n## latency (ms)")
    print("| 지표 | base | 파인튜닝 | Δ |")
    print("|---|---|---|---|")
    for k in ["p50", "p95", "mean"]:
        print(_row(k, b["latency_ms"][k], f["latency_ms"][k], fmt="{:.1f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "finetune_config.yaml"))
    ap.add_argument("--test", default=None, help="test jsonl 경로 (기본: config eval.test_path)")
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 경로(없으면 base)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "FT_JSON"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    ecfg = cfg.get("eval", {})

    def _abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))

    test_path = _abs(args.test or ecfg.get("test_path", "finetune/data/test.jsonl"))
    results_dir = _abs(ecfg.get("results_dir", "finetune/outputs/eval"))
    warmup = args.warmup if args.warmup is not None else int(ecfg.get("warmup", 1))
    adapter = _abs(args.adapter) if args.adapter else None

    run_eval(cfg, repo_root, test_path, adapter, args.tag, warmup, results_dir)


if __name__ == "__main__":
    main()
