"""build_sft_dataset.py — QLoRA SFT 데이터 빌더 (논문급, 누수 차단).

핵심 설계
---------
1) 누수 차단: base 쿼리 단위로 train/val/test 분할.
   - test = held-out 원본 쿼리만(RAG-input 스키마). 학습에 일절 사용 안 함.
   - 패러프레이즈/합성쿼리는 train 에만. val/test 쿼리와 토큰 자카드가 높은
     합성쿼리는 폐기(누수 방지).
2) 규모 확대: 코퍼스 기반 신규 쿼리 합성(teacher) + 패러프레이즈.
3) 품질: 형식·인용 검증(정규화) + 길이/중복 필터.
4) 재현성: dataset_manifest.json(해시/시드/모델/분할/카운트/git commit).

산출
----
- train/val : {"messages":[...]} (SFT, 마지막 turn=assistant 정답)
- test      : RAG-input 스키마(query, retrieved_chunks, relevant_chunk_ids, ground_truth_answer)
- manifest  : 재현 정보

실행
----
  python finetune/build_sft_dataset.py --config finetune/finetune_config.yaml
  (--limit N: base 쿼리 N개로 제한, --dry-run: teacher 호출 없이 점검)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
import prompt_builder  # noqa: E402
import output_parser as op  # noqa: E402


# ─────────────────────────────────────────────────────────────
# teacher LLM 클라이언트 (OpenAI 호환)
# ─────────────────────────────────────────────────────────────
_PROVIDER_BASE_URL = {"upstage": "https://api.upstage.ai/v1", "openai": None}


class TeacherClient:
    def __init__(self, tcfg: dict):
        from openai import OpenAI

        api_key = os.environ.get(tcfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"teacher API 키 환경변수 '{tcfg['api_key_env']}' 가 비어 있습니다.")
        base_url = tcfg.get("base_url") or _PROVIDER_BASE_URL.get(tcfg["provider"])
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = tcfg["model"]
        self.temperature = float(tcfg["temperature"])
        self.max_tokens = int(tcfg["max_tokens"])
        self.timeout = float(tcfg["request_timeout_s"])

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout,
        )
        return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# teacher 지시 프롬프트
# ─────────────────────────────────────────────────────────────
def _teacher_messages(query: str, used_chunks: list[dict], abstain: bool) -> list[dict]:
    context_block = prompt_builder.build_raw_context_block(used_chunks)
    if abstain:
        goal = ("아래 [Retrieved Context]에는 질의의 정답 근거가 없다. "
                "추측하지 말고 반드시 '근거 부족'으로 답하는 정답 예시를 만들어라.")
    else:
        goal = ("아래 [Retrieved Context]만 근거로, 질의에 대한 정확한 정답 예시를 만들어라. "
                "각 주장 문장 끝에 실제 사용한 [Chunk ID]를 표기하고, "
                "<<<EVIDENCE>>>에는 본문에서 인용한 ID의 합집합만 적어라. "
                "컨텍스트에 없는 내용은 추가하지 마라.")
    return [
        {"role": "system", "content": prompt_builder.SYSTEM_INSTRUCTION},
        {"role": "user", "content": (
            f"{goal}\n\n[User Query]\n{query}\n\n{context_block}\n\n"
            "반드시 [Output Format]의 구분자(<<<ANSWER>>>/<<<EVIDENCE>>>/"
            "<<<GROUNDEDNESS_NOTE>>>)를 정확히 지켜 출력만 반환하라.")},
    ]


def _paraphrase_messages(query: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": "너는 한국어 질의 패러프레이즈 생성기다. 의미는 동일하게 유지한다."},
        {"role": "user", "content": (
            f"다음 질문을 의미가 같은 서로 다른 표현 {n}개로 바꿔라. "
            f"번호/설명 없이 한 줄에 하나씩만 출력하라.\n\n질문: {query}")},
    ]


def _synth_query_messages(chunks: list[dict]) -> list[dict]:
    ctx = prompt_builder.build_raw_context_block(chunks)
    return [
        {"role": "system", "content": "너는 한국어 RAG 평가용 질문 생성기다. 주어진 문서로만 답할 수 있는 질문을 만든다."},
        {"role": "user", "content": (
            "다음 문서 청크들만으로 답할 수 있는 자연스러운 한국어 질문 1개만 출력하라. "
            f"설명/번호/따옴표 없이 질문 문장만 출력하라.\n\n{ctx}")},
    ]


# ─────────────────────────────────────────────────────────────
# 타깃 정규화 (canonicalize) — 실제 used_chunk_ids 기준
# ─────────────────────────────────────────────────────────────
def _extract_inline_ids(answer_text: str, used_ids: list[str]) -> list[str]:
    if not answer_text:
        return []
    bracketed = re.findall(r"\[([^\[\]]+)\]", answer_text)
    used_set = set(used_ids)
    seen, ordered = set(), []
    for tok in bracketed:
        t = tok.strip()
        if t in used_set and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _canonicalize_target(raw: str, used_chunks: list[dict], abstain: bool,
                         qmin: int, qmax: int) -> str | None:
    used_ids = [c["chunk_id"] for c in used_chunks]
    answer = op._extract_block(raw, op.ANSWER_DELIM, [op.EVIDENCE_DELIM, op.NOTE_DELIM])
    note_raw = op._extract_block(raw, op.NOTE_DELIM, [op.ANSWER_DELIM, op.EVIDENCE_DELIM])
    if not answer:
        return None
    if not (qmin <= len(answer.strip()) <= qmax):  # 길이 품질 필터
        return None

    if abstain:
        note, evidence_lines = "근거 부족", ["- (해당 없음)"]
    else:
        cited = _extract_inline_ids(answer, used_ids)
        if not cited:
            return None
        note = op._normalize_note(note_raw) or "제공된 근거 문서에 기반함"
        if note == "근거 부족":
            note = "제공된 근거 문서에 기반함"
        evidence_lines = [f"- Chunk ID: {cid}" for cid in cited]

    return (f"{op.ANSWER_DELIM}\n{answer.strip()}\n\n"
            f"{op.EVIDENCE_DELIM}\n" + "\n".join(evidence_lines) + "\n\n"
            f"{op.NOTE_DELIM}\n{note}")


def _to_training_record(query: str, used_chunks: list[dict], target: str,
                        include_fewshot: bool) -> dict:
    if include_fewshot:
        msgs = prompt_builder.build_messages(query, used_chunks)
    else:
        msgs = [
            {"role": "system", "content": prompt_builder.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt_builder.build_user_message(query, used_chunks)},
        ]
    msgs.append({"role": "assistant", "content": target})
    return {"messages": msgs}


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def _norm_tokens(s: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", " ", s.lower()).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _used_from_row(row: dict) -> list[dict]:
    return [{"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
             "content": c["content"], "source_id": c.get("source_id")}
            for c in row["retrieved_chunks"] if c.get("content", "").strip()]


def _generate_valid_target(teacher, query, used_chunks, abstain, max_retries, qmin, qmax):
    for attempt in range(max_retries):
        try:
            out = teacher.chat(_teacher_messages(query, used_chunks, abstain))
        except Exception as e:
            print(f"    [retry {attempt}] teacher API 오류: {e}")
            time.sleep(2.0)
            continue
        tgt = _canonicalize_target(out, used_chunks, abstain, qmin, qmax)
        if tgt:
            return tgt
        print(f"    [retry {attempt}] 정규화/품질 실패")
    return None


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _abspath(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit(repo_root: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _dedup(records: list[dict]) -> list[dict]:
    """정규화 텍스트(user+assistant) 기준 중복 제거."""
    seen, out = set(), []
    for r in records:
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        a = r["messages"][-1]["content"]
        key = re.sub(r"\s+", " ", (u + "||" + a)).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "finetune_config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="base 쿼리 수 상한(스모크)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    s = cfg["synth"]

    eval_rows = _load_jsonl(_abspath(repo_root, s["source"]["manual_eval_set"]))
    corpus = _load_jsonl(_abspath(repo_root, s["source"]["corpus"]))
    if args.limit and args.limit > 0:
        eval_rows = eval_rows[:args.limit]
        print(f"[limit] base 쿼리 {args.limit}건으로 제한")
    print(f"[load] manual eval={len(eval_rows)}건, corpus={len(corpus)}청크")

    # ── 누수 없는 분할 (base 쿼리 단위, 계층 분할) ──
    # 답불가(abstain; relevant 비었거나 X*)와 답가능을 따로 나눠, test/val 에도
    # abstain 커버리지를 보장한다(소수 X쿼리가 한쪽에 몰리는 것 방지).
    sp = s["split"]
    rng = random.Random(sp["seed"])
    rows_by_id = {r["query_id"]: r for r in eval_rows}

    def _is_abs_id(qid):
        r = rows_by_id[qid]
        return (not r.get("relevant_chunk_ids")) or str(qid).upper().startswith("X")

    abs_ids = [q for q in rows_by_id if _is_abs_id(q)]
    ans_ids = [q for q in rows_by_id if not _is_abs_id(q)]
    rng.shuffle(abs_ids)
    rng.shuffle(ans_ids)

    def _split(ids):
        n = len(ids)
        a = int(n * sp["train_ratio"])
        b = a + int(n * sp["val_ratio"])
        return set(ids[:a]), set(ids[a:b]), set(ids[b:])

    tr_a, va_a, te_a = _split(ans_ids)
    # abstain 은 수가 적으므로 test→val→train 순으로 최소 1개씩 보장 배분
    tr_x, va_x, te_x = set(), set(), set()
    for i, qid in enumerate(abs_ids):
        (te_x if i == 0 else va_x if i == 1 else tr_x).add(qid)

    train_ids, val_ids, test_ids = tr_a | tr_x, va_a | va_x, te_a | te_x
    print(f"[split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
          f"(answerable {len(ans_ids)} / abstain {len(abs_ids)}, by query_id)")

    out = s["output"]
    train_path = _abspath(repo_root, out["train_path"])
    val_path = _abspath(repo_root, out["val_path"])
    test_path = _abspath(repo_root, out["test_path"])
    manifest_path = _abspath(repo_root, out["manifest_path"])

    if args.dry_run:
        print(f"[dry-run] 정상. 출력 예정:\n  {train_path}\n  {val_path}\n  {test_path}")
        return

    # ── TEST: held-out 원본만 (학습에 미사용) ──
    test_rows = []
    for qid in test_ids:
        r = rows_by_id[qid]
        test_rows.append({
            "query_id": qid, "query": r["query"],
            "retrieved_chunks": r["retrieved_chunks"],
            "relevant_chunk_ids": r.get("relevant_chunk_ids", []),
            "ground_truth_answer": r.get("ground_truth_answer", ""),
        })
    _write_jsonl(test_path, test_rows)
    print(f"[test] held-out {len(test_rows)}건 저장")

    teacher = TeacherClient(s["teacher"])
    aug, q = s["augment"], s["quality"]
    mr = int(s["teacher"]["max_retries"])
    qmin, qmax = int(q["min_answer_chars"]), int(q["max_answer_chars"])
    fewshot = bool(out["include_fewshot"])

    # 누수 가드용: val/test 쿼리 토큰 집합
    holdout_tok = [_norm_tokens(rows_by_id[i]["query"]) for i in (val_ids | test_ids)]

    def add_answerable(qid_set, n_para, bucket):
        for qid in qid_set:
            r = rows_by_id[qid]
            used = _used_from_row(r)
            if not used:
                continue
            variants = [r["query"]]
            if n_para > 0:
                try:
                    para = teacher.chat(_paraphrase_messages(r["query"], n_para))
                    variants += [ln.strip() for ln in para.splitlines() if ln.strip()][:n_para]
                except Exception as e:
                    print(f"  [{qid}] 패러프레이즈 실패: {e}")
            for qv in variants:
                tgt = _generate_valid_target(teacher, qv, used, False, mr, qmin, qmax)
                if tgt:
                    bucket.append(_to_training_record(qv, used, tgt, fewshot))

    def add_abstain(qid_set, bucket):
        k = int(aug["abstain_distractor_k"])
        for qid in qid_set:
            r = rows_by_id[qid]
            relevant = set(r.get("relevant_chunk_ids", []))
            pool = [c for c in corpus
                    if c["chunk_id"] not in relevant and c.get("content", "").strip()]
            if len(pool) < k:
                continue
            used = [{"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
                     "content": c["content"], "source_id": c.get("source_id")}
                    for c in rng.sample(pool, k)]
            tgt = _generate_valid_target(teacher, r["query"], used, True, mr, qmin, qmax)
            if tgt:
                bucket.append(_to_training_record(r["query"], used, tgt, fewshot))

    train_rec, val_rec = [], []
    smoke = bool(args.limit and args.limit > 0)   # 스모크 모드: 호출 수 최소화

    # ── TRAIN: 원본+패러프레이즈, abstain ──
    print("[train] 답가능 생성...")
    n_para_train = 1 if smoke else int(aug["paraphrases_per_train_query"])
    add_answerable(train_ids, n_para_train, train_rec)
    if aug["make_abstain_examples"]:
        add_abstain(train_ids, train_rec)
        print(f"[train] abstain 후 {len(train_rec)}건")

    # ── TRAIN: 코퍼스 기반 신규 쿼리 합성 (누수 가드) ──
    n_synth = min(3, int(aug["synth_queries_from_corpus"])) if smoke else int(aug["synth_queries_from_corpus"])
    if n_synth > 0:
        kc = int(aug["synth_chunks_per_query"])
        thr = float(aug["leak_jaccard_threshold"])
        text_pool = [c for c in corpus if c.get("content", "").strip()]
        made, tries = 0, 0
        while made < n_synth and tries < n_synth * 4:
            tries += 1
            if len(text_pool) < kc:
                break
            chunks = rng.sample(text_pool, kc)
            used = [{"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
                     "content": c["content"], "source_id": c.get("source_id")} for c in chunks]
            try:
                synq = teacher.chat(_synth_query_messages(used)).splitlines()[0].strip().strip('"')
            except Exception as e:
                print(f"  [synth] 질문 생성 실패: {e}")
                continue
            if not synq:
                continue
            qt = _norm_tokens(synq)
            if any(_jaccard(qt, h) >= thr for h in holdout_tok):
                continue  # val/test와 유사 → 누수 폐기
            tgt = _generate_valid_target(teacher, synq, used, False, mr, qmin, qmax)
            if tgt:
                train_rec.append(_to_training_record(synq, used, tgt, fewshot))
                made += 1
        print(f"[train] 합성 쿼리 {made}건 추가 (시도 {tries})")

    # ── VAL: 원본만 + abstain (eval_loss용) ──
    print("[val] 생성...")
    add_answerable(val_ids, 0, val_rec)
    if aug["make_abstain_examples"]:
        add_abstain(val_ids, val_rec)

    # ── 품질: 중복 제거 ──
    if q["dedup"]:
        train_rec, val_rec = _dedup(train_rec), _dedup(val_rec)
    rng.shuffle(train_rec)

    _write_jsonl(train_path, train_rec)
    _write_jsonl(val_path, val_rec)

    # ── manifest (재현성) ──
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "teacher_model": s["teacher"]["model"],
        "base_split_seed": sp["seed"],
        "split_query_ids": {"train": sorted(train_ids), "val": sorted(val_ids),
                            "test": sorted(test_ids)},
        "counts": {"train": len(train_rec), "val": len(val_rec), "test": len(test_rows)},
        "augment": aug, "quality": q,
        "sha256": {"train": _sha256(train_path), "val": _sha256(val_path),
                   "test": _sha256(test_path)},
        "leak_controls": ["split_by_query_id", "no_paraphrase_in_val_test",
                          "synth_query_jaccard_filter", "test_is_original_heldout"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] train={len(train_rec)} val={len(val_rec)} test={len(test_rows)}")
    print(f"  {train_path}\n  {val_path}\n  {test_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
