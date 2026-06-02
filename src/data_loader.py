"""data_loader.py — 평가셋 로드 + content/embedding 필터링.

책임:
- eval_set.sample.jsonl 로드 및 최소 스키마 검증
- retrieved_chunks 중 텍스트 content가 있는 chunk만 통과 (M2 Content Filter)
- content 없음 / embedding-only chunk 제외 + 개수 집계
- 모든 chunk에 content 없으면 all_content_missing=True (근거 부족 경로)

규칙(실험계획 §6.1.2):
- embedding 필드는 생성 LLM 입력에 절대 사용하지 않는다(여기서 컨텍스트로 넘기지 않음).
- content는 '비어 있지 않은 텍스트'여야 한다.
"""

from __future__ import annotations

import json
from typing import Any


# 생성 모듈 입력 스키마 필수 필드
REQUIRED_QUERY_FIELDS = ["query_id", "query", "retrieved_chunks"]
REQUIRED_CHUNK_FIELDS = ["chunk_id", "modality"]


def load_eval_set(path: str) -> list[dict]:
    """JSONL 평가셋을 읽어 레코드 리스트로 반환한다."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 실패 (line {line_no}): {e}") from e
            _validate_query_record(obj, line_no)
            records.append(obj)
    if not records:
        raise ValueError(f"평가셋이 비어 있습니다: {path}")
    return records


def _validate_query_record(obj: dict, line_no: int) -> None:
    for field in REQUIRED_QUERY_FIELDS:
        if field not in obj:
            raise ValueError(f"질의 레코드 필수 필드 누락 '{field}' (line {line_no})")
    if not isinstance(obj["retrieved_chunks"], list):
        raise ValueError(f"retrieved_chunks 는 리스트여야 합니다 (line {line_no})")
    for ch in obj["retrieved_chunks"]:
        for field in REQUIRED_CHUNK_FIELDS:
            if field not in ch:
                raise ValueError(
                    f"chunk 필수 필드 누락 '{field}' (query {obj.get('query_id')}, line {line_no})"
                )
    # 평가 라벨은 없을 수 있으므로 기본값 보정
    obj.setdefault("relevant_chunk_ids", [])
    obj.setdefault("ground_truth_answer", "")


def _has_text_content(chunk: dict) -> bool:
    """content가 '비어 있지 않은 텍스트'인지 판단한다."""
    content = chunk.get("content", None)
    return isinstance(content, str) and content.strip() != ""


def filter_chunks(retrieved_chunks: list[dict]) -> dict[str, Any]:
    """content 필터링을 적용해 생성 입력용 chunk만 추린다.

    반환:
      used_chunks                : content(텍스트)가 있는 chunk (embedding 필드 제거)
      retrieved_chunk_ids        : 입력으로 받은 전체 chunk ID
      used_chunk_ids             : 실제 컨텍스트에 포함된 chunk ID
      content_missing_chunk_count: content 없어 제외된 chunk 수
      all_content_missing        : 모든 chunk에 content 없음 여부
    """
    used_chunks = []
    retrieved_chunk_ids = []
    content_missing = 0

    for ch in retrieved_chunks:
        retrieved_chunk_ids.append(ch["chunk_id"])
        if _has_text_content(ch):
            # embedding 등 검색용 필드는 생성 입력에 넣지 않는다 (필요 필드만 복사)
            used_chunks.append(
                {
                    "chunk_id": ch["chunk_id"],
                    "modality": ch.get("modality", "text"),
                    "content": ch["content"].strip(),
                    "source_id": ch.get("source_id"),
                }
            )
        else:
            content_missing += 1

    return {
        "used_chunks": used_chunks,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "used_chunk_ids": [c["chunk_id"] for c in used_chunks],
        "content_missing_chunk_count": content_missing,
        "all_content_missing": len(used_chunks) == 0,
    }
