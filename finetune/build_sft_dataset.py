"""build_sft_dataset.py — QLoRA SFT용 합성 학습셋 생성.

흐름:
  1) manual top-5 평가셋(gold context) 로드.
  2) teacher LLM(OpenAI 호환 API)로 각 쿼리의 '정답 출력'을 RECRAG 형식으로 생성.
     - <<<ANSWER>>>/<<<EVIDENCE>>>/<<<GROUNDEDNESS_NOTE>>> + 인라인 [Cxxx] 인용.
     - output_parser.parse_generation_output 로 형식·인용 검증. 실패 시 max_retries 재생성.
  3) (옵션) 쿼리 패러프레이즈로 증강.
  4) (옵션) 무관 청크만 준 abstain(근거부족) 예시 생성.
  5) prompt_builder 로 학습 messages 구성 후 train/val JSONL({"messages":[...]}) 저장.

산출 JSONL 1행 = {"messages": [{"role","content"}, ...]} (마지막 turn = assistant 정답).
이 형식을 train_qlora.py 가 그대로 학습한다.

실행:
  python finetune/build_sft_dataset.py --config finetune/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import yaml

# src 모듈 재사용 (프롬프트/검증 로직을 추론과 동일하게 유지)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
import prompt_builder  # noqa: E402
import output_parser as op  # noqa: E402
from output_parser import parse_generation_output  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 타깃 정규화 (canonicalize)
# ─────────────────────────────────────────────────────────────
# inference 파서는 [C001] 형식만 인식하지만, 실제 데이터의 chunk_id는
# T01_C1 같은 임의 토큰일 수 있다. 학습 타깃은 '실제 used_chunk_ids' 기준으로
# 직접 파싱·재구성해, 인라인 인용 == Evidence 블록을 보장한다.
import re  # noqa: E402


def _extract_inline_ids(answer_text: str, used_ids: list[str]) -> list[str]:
    """answer 본문의 [id] 중 used_ids에 실제 존재하는 것만 등장 순서대로 반환."""
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


def _canonicalize_target(raw: str, used_chunks: list[dict], abstain: bool) -> str | None:
    """teacher 원문을 형식 보장된 학습 타깃으로 재구성. 불가하면 None."""
    used_ids = [c["chunk_id"] for c in used_chunks]
    answer = op._extract_block(raw, op.ANSWER_DELIM, [op.EVIDENCE_DELIM, op.NOTE_DELIM])
    note_raw = op._extract_block(raw, op.NOTE_DELIM, [op.ANSWER_DELIM, op.EVIDENCE_DELIM])
    if not answer:
        return None

    if abstain:
        cited: list[str] = []
        note = "근거 부족"
        evidence_lines = ["- (해당 없음)"]
    else:
        cited = _extract_inline_ids(answer, used_ids)
        if not cited:
            return None  # 답가능인데 유효 인용이 없으면 학습 타깃으로 부적절
        note = op._normalize_note(note_raw) or "제공된 근거 문서에 기반함"
        if note == "근거 부족":  # 인용이 있는데 근거부족 노트면 보정
            note = "제공된 근거 문서에 기반함"
        evidence_lines = [f"- Chunk ID: {cid}" for cid in cited]

    return (
        f"{op.ANSWER_DELIM}\n{answer.strip()}\n\n"
        f"{op.EVIDENCE_DELIM}\n" + "\n".join(evidence_lines) + "\n\n"
        f"{op.NOTE_DELIM}\n{note}"
    )


# ─────────────────────────────────────────────────────────────
# teacher LLM 클라이언트 (OpenAI 호환)
# ─────────────────────────────────────────────────────────────
_PROVIDER_BASE_URL = {
    "upstage": "https://api.upstage.ai/v1",
    "openai": None,            # openai 기본값 사용
}


class TeacherClient:
    def __init__(self, tcfg: dict):
        from openai import OpenAI

        api_key = os.environ.get(tcfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"teacher API 키 환경변수 '{tcfg['api_key_env']}' 가 비어 있습니다."
            )
        base_url = tcfg.get("base_url") or _PROVIDER_BASE_URL.get(tcfg["provider"])
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = tcfg["model"]
        self.temperature = float(tcfg["temperature"])
        self.max_tokens = int(tcfg["max_tokens"])
        self.timeout = float(tcfg["request_timeout_s"])

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# teacher 지시 프롬프트
# ─────────────────────────────────────────────────────────────
def _teacher_messages(query: str, used_chunks: list[dict], abstain: bool) -> list[dict]:
    """teacher 에게 RECRAG 정답 출력을 생성시키는 지시."""
    context_block = prompt_builder.build_raw_context_block(used_chunks)
    if abstain:
        goal = (
            "아래 [Retrieved Context]에는 질의의 정답 근거가 없다. "
            "추측하지 말고 반드시 '근거 부족'으로 답하는 정답 예시를 만들어라."
        )
    else:
        goal = (
            "아래 [Retrieved Context]만 근거로, 질의에 대한 정확한 정답 예시를 만들어라. "
            "각 주장 문장 끝에 실제 사용한 [Chunk ID]를 표기하고, "
            "<<<EVIDENCE>>>에는 본문에서 인용한 ID의 합집합만 적어라. "
            "컨텍스트에 없는 내용은 추가하지 마라."
        )
    return [
        {"role": "system", "content": prompt_builder.SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"{goal}\n\n"
                f"[User Query]\n{query}\n\n{context_block}\n\n"
                "반드시 [Output Format]의 구분자(<<<ANSWER>>>/<<<EVIDENCE>>>/"
                "<<<GROUNDEDNESS_NOTE>>>)를 정확히 지켜 출력만 반환하라."
            ),
        },
    ]


def _paraphrase_messages(query: str, n: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "너는 한국어 질의 패러프레이즈 생성기다. 의미는 동일하게 유지한다.",
        },
        {
            "role": "user",
            "content": (
                f"다음 질문을 의미가 같은 서로 다른 표현 {n}개로 바꿔라. "
                f"번호/설명 없이 한 줄에 하나씩만 출력하라.\n\n질문: {query}"
            ),
        },
    ]


# ─────────────────────────────────────────────────────────────
# 생성 + 검증
# ─────────────────────────────────────────────────────────────
def _generate_valid_target(
    teacher: TeacherClient,
    query: str,
    used_chunks: list[dict],
    abstain: bool,
    max_retries: int,
) -> str | None:
    """teacher 출력을 형식 보장 타깃으로 정규화해 반환. 실패 시 None."""
    for attempt in range(max_retries):
        try:
            out = teacher.chat(_teacher_messages(query, used_chunks, abstain))
        except Exception as e:  # API 오류는 잠깐 쉬고 재시도
            print(f"    [retry {attempt}] teacher API 오류: {e}")
            time.sleep(2.0)
            continue
        target = _canonicalize_target(out, used_chunks, abstain)
        if target:
            return target
        print(f"    [retry {attempt}] 정규화 실패 (answer/인용 부족)")
    return None


def _to_training_record(query: str, used_chunks: list[dict], target: str,
                        include_fewshot: bool) -> dict:
    """학습용 {"messages":[...]} 레코드. 마지막 turn = assistant 정답."""
    if include_fewshot:
        msgs = prompt_builder.build_messages(query, used_chunks)
    else:
        # few-shot 없이 system -> user 만 (형식은 정답 타깃으로 학습)
        msgs = [
            {"role": "system", "content": prompt_builder.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt_builder.build_user_message(query, used_chunks)},
        ]
    msgs.append({"role": "assistant", "content": target})
    return {"messages": msgs}


# ─────────────────────────────────────────────────────────────
# 입력 로드 헬퍼
# ─────────────────────────────────────────────────────────────
def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _abspath(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "finetune_config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="teacher 호출 없이 입력/경로/개수만 점검")
    ap.add_argument("--limit", type=int, default=0,
                    help="처리할 base 쿼리 수 상한(스모크 테스트용, 0=전체)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    base_dir = os.path.dirname(base_dir)  # repo 루트 (config는 finetune/ 안)

    s = cfg["synth"]
    rng = random.Random(s["augment"]["seed"])

    manual_path = _abspath(base_dir, s["source"]["manual_eval_set"])
    corpus_path = _abspath(base_dir, s["source"]["corpus"])
    eval_rows = _load_jsonl(manual_path)
    corpus = _load_jsonl(corpus_path)
    if args.limit and args.limit > 0:
        eval_rows = eval_rows[:args.limit]
        print(f"[limit] base 쿼리 {args.limit}건으로 제한 (스모크 테스트)")
    print(f"[load] manual eval={len(eval_rows)}건, corpus={len(corpus)}청크")

    train_path = _abspath(base_dir, s["output"]["train_path"])
    val_path = _abspath(base_dir, s["output"]["val_path"])
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] 정상. 출력 예정: {train_path} / {val_path}")
        return

    teacher = TeacherClient(s["teacher"])
    max_retries = int(s["teacher"]["max_retries"])
    n_para = int(s["augment"]["paraphrases_per_query"])
    include_fewshot = bool(s["output"]["include_fewshot"])

    records: list[dict] = []

    # ── 1) 답가능 예시 (gold context) + 패러프레이즈 증강 ──
    for i, row in enumerate(eval_rows):
        query = row["query"]
        used = [
            {"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
             "content": c["content"], "source_id": c.get("source_id")}
            for c in row["retrieved_chunks"] if c.get("content", "").strip()
        ]
        if not used:
            continue

        variants = [query]
        if n_para > 0:
            try:
                para = teacher.chat(_paraphrase_messages(query, n_para))
                variants += [ln.strip() for ln in para.splitlines() if ln.strip()][:n_para]
            except Exception as e:
                print(f"  [{row.get('query_id')}] 패러프레이즈 실패(원문만 사용): {e}")

        for q in variants:
            tgt = _generate_valid_target(teacher, q, used, abstain=False,
                                         max_retries=max_retries)
            if tgt:
                records.append(_to_training_record(q, used, tgt, include_fewshot))
        print(f"[{i+1}/{len(eval_rows)}] {row.get('query_id')} -> 누적 {len(records)}건")

    # ── 2) abstain(근거부족) 예시 ──
    if s["augment"]["make_abstain_examples"]:
        k = int(s["augment"]["abstain_distractor_k"])
        for row in eval_rows:
            relevant = set(row.get("relevant_chunk_ids", []))
            # 정답 근거와 무관한 청크만 distractor 로
            pool = [c for c in corpus
                    if c["chunk_id"] not in relevant and c.get("content", "").strip()]
            if len(pool) < k:
                continue
            distractors = rng.sample(pool, k)
            used = [{"chunk_id": c["chunk_id"], "modality": c.get("modality", "text"),
                     "content": c["content"], "source_id": c.get("source_id")}
                    for c in distractors]
            tgt = _generate_valid_target(teacher, row["query"], used, abstain=True,
                                         max_retries=max_retries)
            if tgt:
                records.append(_to_training_record(row["query"], used, tgt, include_fewshot))
        print(f"[abstain] 추가 후 누적 {len(records)}건")

    # ── 3) train/val 분할 + 저장 ──
    rng.shuffle(records)
    n_val = max(1, int(len(records) * float(s["output"]["val_ratio"])))
    val, train = records[:n_val], records[n_val:]

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] train={len(train)} val={len(val)}")
    print(f"  {train_path}\n  {val_path}")


if __name__ == "__main__":
    main()
