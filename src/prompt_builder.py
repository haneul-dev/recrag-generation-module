"""prompt_builder.py — Raw Context 프롬프트(ChatML messages) 생성.

구성(rag_prompt_template §2~§3, prompt_type=groundedness):
  system  : 공통 System Instruction (제약 9개)
  assistant: few-shot 예시 2개 (정상 1 + 근거부족 1, 한국어) — 형식 앵커
  user    : [User Query] + [Retrieved Context](Raw 블록)

1차 범위: context_type=raw, prompt_type=groundedness 고정.
Raw 블록만 컨텍스트 구성 방식에 해당하며, System/few-shot/Output Format은 공통(비교 공정성).
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# 공통 System Instruction (rag_prompt_template §2.2, 고정)
# ─────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """너는 RECRAG의 생성 모듈이다. 너의 역할은 아래 [Retrieved Context]에 제공된 문서 청크만을
근거로 사용자 질의에 답하는 것이다.

반드시 다음 규칙을 지켜라.
1. [Retrieved Context]에 없는 내용은 답하지 않는다. 너의 사전 지식으로 보충하지 않는다.
2. 확신이 서지 않으면 추측하지 말고 "근거 부족"을 선택한다.
3. 답변의 각 주요 주장 문장 끝에 근거 Chunk ID를 [C001] 형태로 표시한다.
4. 하나의 주장에 여러 근거가 있으면 [C001][C004]처럼 모두 표시한다.
5. <<<EVIDENCE>>>에는 본문에서 실제로 인용한 Chunk ID의 합집합만 나열한다.
   사용하지 않은 청크 ID는 적지 않는다.
6. 청크 간 내용이 모순되면 모순이 있음을 답변에 밝힌다.
7. 텍스트화된 evidence(이미지 캡션/OCR, 오디오 전사/요약)는 제공된 청크의 범위
   안에서만 해석하고, 이미지·오디오 원본을 추정하거나 재해석하지 않는다.
8. 아래 [Output Format]의 형식과 구분자를 정확히 지켜 출력한다.

[Output Format]
<<<ANSWER>>>
(각 주요 주장 문장 끝에 [Chunk ID]를 표시한 자연어 설명)

<<<EVIDENCE>>>
- Chunk ID: (본문에서 실제 인용한 ID들의 합집합. 여러 개면 줄마다 하나씩. 없으면 "- (해당 없음)")

<<<GROUNDEDNESS_NOTE>>>
(제공된 근거 문서에 기반함 / 일부 근거 부족 / 근거 부족 중 하나)"""


# ─────────────────────────────────────────────────────────────
# 공통 Few-shot 예시 (rag_prompt_template §2.3, 한국어, 짧게)
# ─────────────────────────────────────────────────────────────
FEWSHOT_USER_1 = """[User Query]
생성 모듈은 무엇을 출력하는가?

[Retrieved Context]
- Chunk ID: C001
  생성 모듈은 검색된 문서를 근거로 자연어 설명을 생성한다.
- Chunk ID: C004
  생성 답변에는 참조한 청크 ID를 함께 출력한다."""

FEWSHOT_ASSISTANT_1 = """<<<ANSWER>>>
생성 모듈은 검색된 문서를 근거로 자연어 설명을 생성한다. [C001] 답변에는 참조 청크 ID를 함께 출력한다. [C004]

<<<EVIDENCE>>>
- Chunk ID: C001
- Chunk ID: C004

<<<GROUNDEDNESS_NOTE>>>
제공된 근거 문서에 기반함."""

FEWSHOT_USER_2 = """[User Query]
이 시스템의 가격 정책은 무엇인가?

[Retrieved Context]
- Chunk ID: C201
  생성 모듈은 검색 문서를 근거로 설명을 생성한다."""

FEWSHOT_ASSISTANT_2 = """<<<ANSWER>>>
제공된 근거 문서에서 해당 질의에 답할 수 있는 내용을 찾지 못했습니다.

<<<EVIDENCE>>>
- (해당 없음)

<<<GROUNDEDNESS_NOTE>>>
근거 부족."""


def build_raw_context_block(used_chunks: list[dict]) -> str:
    """Raw Context [Retrieved Context] 블록 생성 (rag_prompt_template §3.1).

    원문 청크를 거의 그대로 입력한다. embedding 등 비텍스트 필드는 포함하지 않는다.
    """
    lines = ["[Retrieved Context]"]
    for ch in used_chunks:
        lines.append(f"- Chunk ID: {ch['chunk_id']}")
        # content 는 여러 줄일 수 있으므로 들여쓰기로 이어 붙인다.
        for content_line in ch["content"].splitlines() or [""]:
            lines.append(f"  {content_line}")
    return "\n".join(lines)


def build_user_message(query: str, used_chunks: list[dict]) -> str:
    """user 메시지 전체 구성 (Raw): [User Query] + [Retrieved Context]."""
    context_block = build_raw_context_block(used_chunks)
    return f"[User Query]\n{query}\n\n{context_block}"


def build_messages(query: str, used_chunks: list[dict]) -> list[dict]:
    """ChatML messages 구성: system -> few-shot(user/assistant) -> user.

    apply_chat_template 에 그대로 넘길 수 있는 role/content 리스트를 반환한다.
    """
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": FEWSHOT_USER_1},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_1},
        {"role": "user", "content": FEWSHOT_USER_2},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": build_user_message(query, used_chunks)},
    ]
