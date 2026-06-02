"""output_parser.py — 생성 출력 파싱 (M6 Output Parser).

책임(rag_prompt_template §5, latency_logging_schema §3.3):
- <<<ANSWER>>> / <<<EVIDENCE>>> / <<<GROUNDEDNESS_NOTE>>> 블록 분리
- 본문 인라인 [Cxxx] 파싱 -> inline_cited_chunk_ids (중복 허용 multiset)
- <<<EVIDENCE>>> 파싱 -> evidence_block_chunk_ids
- cited_chunk_ids = set(evidence_block)
- inline_evidence_set_match = (set(inline) == set(evidence_block))
- format_compliance: 세 구분자 존재 + 집합 일치 + cited ⊆ used
"""

from __future__ import annotations

import re


ANSWER_DELIM = "<<<ANSWER>>>"
EVIDENCE_DELIM = "<<<EVIDENCE>>>"
NOTE_DELIM = "<<<GROUNDEDNESS_NOTE>>>"

GROUNDEDNESS_ALLOWED = {"제공된 근거 문서에 기반함", "일부 근거 부족", "근거 부족"}

# 인라인 인용 [C001] / Evidence 줄의 Chunk ID 추출용
_INLINE_RE = re.compile(r"\[(C\d+)\]")
_EVIDENCE_ID_RE = re.compile(r"Chunk\s*ID\s*:\s*(C\d+)", re.IGNORECASE)


def _extract_block(text: str, start_delim: str, next_delims: list[str]) -> str | None:
    """start_delim 다음부터 다음 구분자 전까지의 본문을 반환한다."""
    idx = text.find(start_delim)
    if idx == -1:
        return None
    start = idx + len(start_delim)
    end = len(text)
    for nd in next_delims:
        nidx = text.find(nd, start)
        if nidx != -1:
            end = min(end, nidx)
    return text[start:end].strip()


def _normalize_note(raw_note: str | None) -> str | None:
    """Groundedness Note 를 허용값 중 하나로 정규화한다."""
    if raw_note is None:
        return None
    cleaned = raw_note.strip().rstrip(".。").strip()
    # 정확 일치 우선, 아니면 포함 매칭
    if cleaned in GROUNDEDNESS_ALLOWED:
        return cleaned
    for allowed in GROUNDEDNESS_ALLOWED:
        if allowed in cleaned:
            return allowed
    return None


def parse_generation_output(output_text: str, used_chunk_ids: list[str]) -> dict:
    """생성 텍스트를 구조화 결과로 파싱한다.

    반환:
      answer, inline_cited_chunk_ids, evidence_block_chunk_ids, cited_chunk_ids,
      groundedness_note, inline_evidence_set_match, format_compliance, error_type
    """
    used_set = set(used_chunk_ids)
    error_reasons = []

    answer = _extract_block(output_text, ANSWER_DELIM, [EVIDENCE_DELIM, NOTE_DELIM])
    evidence_raw = _extract_block(output_text, EVIDENCE_DELIM, [NOTE_DELIM, ANSWER_DELIM])
    note_raw = _extract_block(output_text, NOTE_DELIM, [ANSWER_DELIM, EVIDENCE_DELIM])

    # 1) 구분자 존재 여부
    if answer is None:
        error_reasons.append("missing_answer_block")
    if evidence_raw is None:
        error_reasons.append("missing_evidence_block")
    if note_raw is None:
        error_reasons.append("missing_note_block")

    # 2) 인라인 인용 (중복 허용)
    inline_cited = _INLINE_RE.findall(answer or "")

    # 3) Evidence 블록 ID
    evidence_block = _EVIDENCE_ID_RE.findall(evidence_raw or "")

    cited_set = set(evidence_block)
    cited_chunk_ids = sorted(cited_set)

    # 4) 인라인 ↔ Evidence 집합 일치
    inline_evidence_set_match = set(inline_cited) == cited_set
    if not inline_evidence_set_match:
        error_reasons.append("inline_evidence_set_mismatch")

    # 5) Groundedness Note 정규화
    note = _normalize_note(note_raw)
    if note is None:
        error_reasons.append("invalid_groundedness_note")

    # 6) 없는 청크 인용 여부 (cited ⊆ used)
    if not cited_set.issubset(used_set):
        error_reasons.append("cited_not_in_used")

    format_compliance = len(error_reasons) == 0

    return {
        "answer": answer if answer is not None else output_text.strip(),
        "inline_cited_chunk_ids": inline_cited,            # multiset (중복 허용)
        "evidence_block_chunk_ids": evidence_block,
        "cited_chunk_ids": cited_chunk_ids,                # 정규 집합 (정렬)
        "groundedness_note": note,
        "inline_evidence_set_match": inline_evidence_set_match,
        "format_compliance": format_compliance,
        "error_type": ";".join(error_reasons) if error_reasons else None,
    }
