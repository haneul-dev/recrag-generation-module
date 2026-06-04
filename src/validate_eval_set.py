"""validate_eval_set.py — 본 실행 전 eval set 스키마·라벨 품질 점검.

생성 모듈 Step 8~9 입력 데이터 검증 범위만 처리한다.
(검색·재정렬·임베딩·HNSW 미수정. 평가 채점 로직 아님 — 입력 품질 점검만.)

사용:
  # B안: standalone
  python src/validate_eval_set.py --eval-set data/eval_set.interim.jsonl

  # A안: run_generation 에서 --validate-only 로 동일 로직 호출
  python src/run_generation.py --eval-set data/eval_set.interim.jsonl --validate-only

종료 코드: FAIL 이면 1, PASS/WARN 이면 0.
"""

from __future__ import annotations

import argparse
import json
import sys


ALLOWED_MODALITIES = {
    "text", "image_caption", "image_ocr", "audio_transcript", "audio_summary",
}
RECOMMENDED_CHUNKS = 5  # config top_k_input 기준 권장값

# abstain(정당한 기권) 라벨로 인정할 GT 문구 키워드
_ABSTAIN_KEYWORDS = ("근거", "없", "부족", "찾지 못", "찾을 수 없")


def _is_abstain_gt(gt: str) -> bool:
    """ground_truth_answer 가 abstain 계열 문구인지(느슨한) 판정."""
    if not gt:
        return False
    g = gt.strip()
    return ("근거" in g) and any(k in g for k in ("없", "부족", "찾지 못", "찾을 수 없"))


def validate_eval_set(path: str) -> dict:
    """eval set 을 점검하고 결과 dict 를 반환한다(예외를 던지지 않고 수집)."""
    errors: list[str] = []   # FAIL
    warnings: list[str] = []  # WARN
    infos: list[str] = []     # INFO

    # ── 파일 로드 (라인별, 예외 수집) ──
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append((ln, json.loads(s)))
                except json.JSONDecodeError as e:
                    errors.append(f"[line {ln}] JSON 파싱 실패: {e}")
    except FileNotFoundError:
        return {
            "status": "FAIL", "n_rows": 0,
            "errors": [f"파일을 찾을 수 없습니다: {path}"],
            "warnings": [], "infos": [], "counts": {},
        }

    n_rows = len(rows)

    counts = {
        "n_rows": n_rows,
        "abstain_cases": 0,
        "queries_all_content_empty": 0,
        "empty_content_chunks": 0,
        "queries_with_embedding": 0,
        "queries_empty_gt": 0,
        "queries_empty_relevant": 0,
        "queries_chunk_count_not_5": 0,
    }

    # 1. 전체 row 수
    if n_rows == 0:
        errors.append("유효한 row 가 0건입니다(빈 파일).")

    seen_qids: dict[str, int] = {}

    for ln, obj in rows:
        qid = obj.get("query_id", f"<no-id line{ln}>")

        # 2. query_id 중복
        if "query_id" in obj:
            if obj["query_id"] in seen_qids:
                errors.append(
                    f"[line {ln}] query_id 중복: '{obj['query_id']}' "
                    f"(먼저 line {seen_qids[obj['query_id']]})"
                )
            else:
                seen_qids[obj["query_id"]] = ln

        # 3. 질의 필수 필드
        for fld in ("query_id", "query", "retrieved_chunks",
                    "relevant_chunk_ids", "ground_truth_answer"):
            if fld not in obj:
                if fld in ("relevant_chunk_ids", "ground_truth_answer"):
                    warnings.append(f"[{qid}] 평가용 필드 누락: '{fld}' (평가 불가)")
                else:
                    errors.append(f"[{qid}] 필수 필드 누락: '{fld}'")

        chunks = obj.get("retrieved_chunks")

        # 4. retrieved_chunks 리스트 여부
        if not isinstance(chunks, list):
            errors.append(f"[{qid}] retrieved_chunks 가 리스트가 아님")
            continue

        chunk_ids = []
        empty_content_in_q = 0
        for i, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                errors.append(f"[{qid}] chunk[{i}] 가 객체(dict)가 아님")
                continue
            cid = ch.get("chunk_id", f"<no-cid idx{i}>")
            chunk_ids.append(ch.get("chunk_id"))

            # 5. chunk 필수 필드
            if "chunk_id" not in ch:
                errors.append(f"[{qid}] chunk[{i}] chunk_id 누락")
            if "modality" not in ch:
                errors.append(f"[{qid}] chunk {cid} modality 누락")
            else:
                # 12. modality 허용값
                if ch["modality"] not in ALLOWED_MODALITIES:
                    errors.append(
                        f"[{qid}] chunk {cid} modality 허용값 아님: '{ch['modality']}' "
                        f"(허용: {sorted(ALLOWED_MODALITIES)})"
                    )
            if "content" not in ch:
                warnings.append(f"[{qid}] chunk {cid} content 키 없음(생성에서 제외됨)")
                empty_content_in_q += 1
            elif not isinstance(ch["content"], str) or ch["content"].strip() == "":
                # 7. content 빈 문자열
                empty_content_in_q += 1
            if "source_id" not in ch:
                warnings.append(f"[{qid}] chunk {cid} source_id 없음(선택, 권장)")

            # 11. embedding 포함 경고
            if "embedding" in ch:
                counts["queries_with_embedding"] += 1
                warnings.append(
                    f"[{qid}] chunk {cid} 에 embedding 포함 — 생성 입력 미사용. "
                    f"용량 절감 위해 제거 권장"
                )

        counts["empty_content_chunks"] += empty_content_in_q

        # 6. chunk 개수 (권장 5)
        n_ch = len(chunks)
        rel = obj.get("relevant_chunk_ids", [])
        gt = obj.get("ground_truth_answer", "") or ""
        is_abstain = (isinstance(rel, list) and len(rel) == 0) or _is_abstain_gt(gt)
        if n_ch != RECOMMENDED_CHUNKS:
            counts["queries_chunk_count_not_5"] += 1
            note = " (abstain/예외 케이스로 보임 — 허용)" if is_abstain else ""
            warnings.append(
                f"[{qid}] retrieved_chunks 개수 {n_ch} (권장 {RECOMMENDED_CHUNKS}){note}"
            )

        # 8. content 전부 비어있는 query
        if n_ch > 0 and empty_content_in_q == n_ch:
            counts["queries_all_content_empty"] += 1
            if is_abstain:
                infos.append(f"[{qid}] 모든 chunk content 비어있음 → 런타임 deterministic abstain")
            else:
                warnings.append(
                    f"[{qid}] 모든 chunk content 비어있는데 relevant/GT 는 답변형 "
                    f"— 런타임에 abstain 처리됨(불일치 점검 필요)"
                )

        # 9. relevant_chunk_ids 가 retrieved chunk_id 안에 있는지
        if isinstance(rel, list):
            present = set(c for c in chunk_ids if c is not None)
            for rid in rel:
                if rid not in present:
                    errors.append(
                        f"[{qid}] relevant_chunk_id '{rid}' 가 retrieved_chunks 에 없음"
                    )
            # 14. relevant 과다/공백
            if len(rel) == 0:
                counts["queries_empty_relevant"] += 1
            elif len(rel) > n_ch:
                warnings.append(
                    f"[{qid}] relevant_chunk_ids({len(rel)}) 가 chunk 수({n_ch})보다 많음"
                )

        # 10. ground_truth_answer 공백
        if gt.strip() == "":
            counts["queries_empty_gt"] += 1
            warnings.append(f"[{qid}] ground_truth_answer 가 비어 있음(평가 불가)")

        # 13. abstain 케이스 카운트 + GT 문구 점검
        if isinstance(rel, list) and len(rel) == 0:
            counts["abstain_cases"] += 1
            if gt.strip() and not _is_abstain_gt(gt):
                warnings.append(
                    f"[{qid}] relevant 가 비어있으나(abstain) GT 가 abstain 문구 계열 아님 "
                    f"— GT 확인 권장"
                )

    # 14(전체). 모든 query 의 relevant 가 비어 있으면 경고
    if n_rows > 0 and counts["queries_empty_relevant"] == n_rows:
        warnings.append(
            "모든 query 의 relevant_chunk_ids 가 비어 있음 — "
            "Source Attribution 평가가 전부 불가(전부 abstain 셋?)"
        )

    # 권장 건수 안내
    if 0 < n_rows < 30:
        warnings.append(f"row 수 {n_rows}건 — 본 실험 권장 최소 30건 미만(interim 예비는 허용)")

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {
        "status": status, "n_rows": n_rows,
        "errors": errors, "warnings": warnings, "infos": infos, "counts": counts,
    }


def print_report(result: dict, path: str) -> None:
    print("=" * 60)
    print(f"eval set 검증: {path}")
    print("=" * 60)
    c = result["counts"]
    print(f"[1] 전체 row 수           : {result['n_rows']}")
    if c:
        print(f"    abstain 케이스        : {c.get('abstain_cases')}")
        print(f"    content 빈 chunk 수   : {c.get('empty_content_chunks')}")
        print(f"    content 전무 query    : {c.get('queries_all_content_empty')}")
        print(f"    chunk수≠5 query       : {c.get('queries_chunk_count_not_5')}")
        print(f"    GT 빈 query           : {c.get('queries_empty_gt')}")
        print(f"    relevant 빈 query     : {c.get('queries_empty_relevant')}")
        print(f"    embedding 포함 chunk  : {c.get('queries_with_embedding')}")

    if result["infos"]:
        print(f"\n[INFO] {len(result['infos'])}건")
        for m in result["infos"]:
            print(f"  · {m}")
    if result["warnings"]:
        print(f"\n[WARN] {len(result['warnings'])}건")
        for m in result["warnings"]:
            print(f"  ! {m}")
    if result["errors"]:
        print(f"\n[FAIL] {len(result['errors'])}건")
        for m in result["errors"]:
            print(f"  ✗ {m}")

    print("\n" + "-" * 60)
    print(f"[15] 검증 결과: {result['status']}  "
          f"(errors={len(result['errors'])}, warnings={len(result['warnings'])})")
    if result["status"] == "FAIL":
        print("     → FAIL 항목을 수정한 뒤 본 실행하세요.")
    elif result["status"] == "WARN":
        print("     → 경고는 검토 후 진행 가능(interim 예비 실험은 대개 허용).")
    else:
        print("     → 통과. 본 실행 가능.")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="RECRAG 생성 모듈 eval set 검증")
    parser.add_argument("--eval-set", required=True, help="검증할 eval set jsonl 경로")
    args = parser.parse_args()
    result = validate_eval_set(args.eval_set)
    print_report(result, args.eval_set)
    sys.exit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
