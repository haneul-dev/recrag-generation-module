"""build_synth_test.py — 합성 held-out 평가셋 확대 (통계력 보강).

목적
----
gold test(원본 30쿼리 분할)는 n이 작아 통계적 검정력이 약하다. 코퍼스에서
**학습/기존 test와 겹치지 않는** 신규 쿼리를 합성해 평가셋을 키운다.

주의: 합성 라벨은 **약지도(weak supervision)** 다. gold test 와 **분리 보고**하고,
일부는 사람 검수를 권장한다(각 행에 provenance="synthetic" 표기).

산출(RAG-input 스키마, evaluate.py 로 평가 가능):
  {query_id, query, retrieved_chunks[...], relevant_chunk_ids, ground_truth_answer,
   provenance:"synthetic"}

누수 차단:
  - 합성 쿼리가 train/val/test(gold) 쿼리와 토큰 자카드 ≥ 임계면 폐기.

실행:
  python finetune/build_synth_test.py --config finetune/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import build_sft_dataset as B  # 헬퍼/클라이언트 재사용  # noqa: E402


def _gold_answer_messages(query: str, chunks: list[dict]) -> list[dict]:
    import prompt_builder
    ctx = prompt_builder.build_raw_context_block(chunks)
    return [
        {"role": "system", "content": "너는 한국어 RAG 정답 작성기다. 주어진 문서 근거로만 간결히 답한다."},
        {"role": "user", "content": (
            f"다음 문서만 근거로 질문에 1~3문장으로 답하라. 형식기호/인용표기 없이 "
            f"평서문 정답만 출력하라.\n\n[질문]\n{query}\n\n{ctx}")},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "finetune_config.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="스모크: 생성 수 상한")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    st = cfg.get("synth_test", {})
    s = cfg["synth"]

    corpus = B._load_jsonl(B._abspath(repo_root, s["source"]["corpus"]))
    out_path = B._abspath(repo_root, cfg["eval"]["test_synth_path"])

    # 누수 가드 대상: gold test + 기존 train/val 쿼리 텍스트
    holdout_tok = []
    gold_test = B._abspath(repo_root, s["output"]["test_path"])
    if os.path.exists(gold_test):
        for r in B._load_jsonl(gold_test):
            holdout_tok.append(B._norm_tokens(r["query"]))
    train_p = B._abspath(repo_root, s["output"]["train_path"])
    if os.path.exists(train_p):
        for r in B._load_jsonl(train_p):
            u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            # "[User Query]\n{q}\n\n..." 에서 쿼리 추출
            q = u.split("[User Query]")[-1].split("[Retrieved Context]")[0].strip()
            if q:
                holdout_tok.append(B._norm_tokens(q))

    count = int(st.get("count", 30))
    if args.limit:
        count = min(count, args.limit)
    kc = int(st.get("chunks_per_query", 2))
    nd = int(st.get("distractors", 3))
    thr = float(st.get("leak_jaccard_threshold", 0.6))
    rng = random.Random(int(s["split"]["seed"]) + 7)

    teacher = B.TeacherClient(s["teacher"])
    text_pool = [c for c in corpus if c.get("content", "").strip()]

    rows, made, tries = [], 0, 0
    while made < count and tries < count * 5:
        tries += 1
        if len(text_pool) < kc + nd:
            break
        relevant = rng.sample(text_pool, kc)
        rel_ids = [c["chunk_id"] for c in relevant]
        try:
            q = teacher.chat(B._synth_query_messages(
                [{"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
                  "content": c["content"], "source_id": c.get("source_id")} for c in relevant]
            )).splitlines()[0].strip().strip('"')
        except Exception as e:
            print(f"  [질문 실패] {e}")
            continue
        if not q:
            continue
        qt = B._norm_tokens(q)
        if any(B._jaccard(qt, h) >= thr for h in holdout_tok):
            continue  # 누수 폐기
        try:
            gt = teacher.chat(_gold_answer_messages(q, relevant)).strip()
        except Exception as e:
            print(f"  [정답 실패] {e}")
            continue
        if not gt:
            continue
        # distractor 포함 retrieved_chunks 구성(인용 변별 테스트)
        pool = [c for c in text_pool if c["chunk_id"] not in set(rel_ids)]
        distractors = rng.sample(pool, min(nd, len(pool)))
        retrieved = relevant + distractors
        rng.shuffle(retrieved)
        rows.append({
            "query_id": f"S{made+1:03d}", "query": q,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
                 "content": c["content"], "source_id": c.get("source_id")} for c in retrieved],
            "relevant_chunk_ids": rel_ids,
            "ground_truth_answer": gt,
            "provenance": "synthetic",
        })
        holdout_tok.append(qt)  # 합성끼리도 중복 방지
        made += 1
        if made % 10 == 0:
            print(f"  [{made}/{count}] 생성")

    B._write_jsonl(out_path, rows)
    print(f"[done] 합성 test {len(rows)}건 (시도 {tries}) -> {out_path}")
    print("주의: 약지도 라벨. gold test와 분리 보고하고 일부는 사람 검수 권장.")


if __name__ == "__main__":
    main()
