# RECRAG Step 8~9 생성 모듈 — Raw Context Baseline

RAG 파이프라인에서 **검색·재정렬로 선별된 텍스트 evidence**를 입력받아,
**Qwen2.5-7B-Instruct**가 근거 기반 자연어 설명 + 참조 Chunk ID + Groundedness Note를
생성하고, **generation latency와 출력 결과를 저장**하는 모듈의 1차 최소 실행 버전이다.

---

## 1. 프로젝트 목적

- **RECRAG Step 8~9 생성 모듈의 Raw Context Baseline** 구현.
- 앞단 검색/전처리 모듈이 넘겨주는 `retrieved_chunks` JSONL(고정 스냅샷)을 입력으로 받아,
  Qwen2.5-7B-Instruct로 답변을 생성한다. (검색은 직접 호출/구현하지 않는다.)
- 답변마다 **참조 Chunk ID(인라인 + Evidence)** 와 **Groundedness Note**를 함께 출력한다.
- **generation latency(t0~t5)** 와 토큰 수, 출력 결과를 JSONL/CSV로 저장한다.

1차 고정값: `context_type=raw` / `prompt_type=groundedness` / `top_k=5` / `query_language=ko` /
`repeats=3` / `warmup=1`.

---

## 2. 담당 범위

| 구분 | 내용 |
|---|---|
| **포함** | 생성 모듈, Raw Context 프롬프트, Qwen 추론, latency logging, output parsing |
| **제외** | 검색, 재정렬, 임베딩, HNSW, 이미지/오디오 **원본** 처리, Structured Context, 평가자 채점, LLM-as-Judge |

> 검색·재정렬 결과는 **고정 스냅샷 파일(JSONL)** 로 입력받는다고 가정한다.
> 이미지·오디오는 선행 전처리에서 **텍스트화된 evidence**로만 들어오며, 생성 모듈은 그 텍스트만 사용한다.

---

## 3. 입력 데이터 형식

`data/eval_set.sample.jsonl` — 1줄 = 1개 질의(JSON).

```json
{
  "query_id": "Q001",
  "query": "RECRAG 생성 모듈의 역할은 무엇인가?",
  "retrieved_chunks": [
    {
      "chunk_id": "C001",
      "modality": "text",
      "content": "LLM이 읽을 수 있는 텍스트 evidence (필수)",
      "source_id": "DOC_RECRAG_01"
    },
    {
      "chunk_id": "C014",
      "modality": "image_caption",
      "content": "",
      "embedding": [0.12, -0.04, 0.33, 0.08]
    }
  ],
  "relevant_chunk_ids": ["C001"],
  "ground_truth_answer": "정답 설명(평가셋 라벨)"
}
```

| 필드 | 필수/선택 | 규칙 |
|---|---|---|
| `chunk_id`, `modality` | 필수 | — |
| `content` | **필수(텍스트 evidence)** | 비어 있거나 없으면 해당 chunk는 **컨텍스트에서 제외** |
| `source_id` | 선택 | — |
| `embedding` | 선택 | **생성 입력에 사용하지 않음**(검색/재현성용). LLM 프롬프트에 절대 포함되지 않음 |
| `relevant_chunk_ids`, `ground_truth_answer` | 선택(평가용) | 없으면 빈값으로 보정 |

**처리 규칙**
- `content` 있는 chunk만 컨텍스트에 포함. 없으면 제외하고 `content_missing_chunk_count`에 집계.
- 한 질의의 **모든** chunk에 content가 없으면 → LLM 호출 없이 **근거 부족**으로 처리(`error_type=all_content_missing`).

샘플 4건 구성:

| query_id | 케이스 |
|---|---|
| Q001 | 정상 text chunk |
| Q002 | image_caption / image_ocr (+ embedding-only C014 제외 확인) |
| Q003 | audio_transcript / audio_summary |
| Q004 | content 전부 없음 → 근거 부족 처리 |

---

## 4. Colab 실행 순서

> 명령어 전체 흐름은 [`colab_run_commands.md`](./colab_run_commands.md) 참고.

```bash
# 1) GPU 확인
!nvidia-smi

# 2) repo clone
!git clone <YOUR_GITHUB_REPO_URL>
%cd recrag_generation

# 3) 패키지 설치
!pip install -r requirements.txt

# 4) 실행
!python src/run_generation.py --config generation_experiment_config.yaml

# 5) 결과 확인
!ls outputs
!head -n 5 outputs/generation_results.csv
```

VRAM이 부족하면 `generation_experiment_config.yaml` 의 `model.quantization` 을
`4bit` 또는 `8bit` 로 설정한다.

### Hugging Face 로그인 (필요 시)
Qwen2.5-7B-Instruct는 공개 모델이라 보통 로그인 없이 받아지지만, 다운로드가 막히면:
```python
from huggingface_hub import login
login()   # 입력창에 토큰을 붙여넣는다
```
> ⚠️ **토큰/API 키/Colab 인증정보를 코드·config·노트북 셀에 직접 적지 말 것.**
> `login()` 입력창에만 입력하고, repo 어디에도 저장하지 않는다. (`.gitignore`로도 차단)

---

## 5. 출력 파일 설명

| 파일 | 설명 |
|---|---|
| `outputs/generation_results.jsonl` | **1차 저장**. 1행 = (query × run) 1건. 타임스탬프·인용·latency 전 필드 포함 |
| `outputs/generation_results.csv` | 분석용 파생(평탄화). 배열 컬럼은 `"C001;C004"` 형태로 직렬화 |

> `outputs/` 는 `.gitignore` 대상이라 repo에 올라가지 않으며, 실행 시 자동 생성된다.

---

## 6. 주요 결과 컬럼 설명

| 컬럼 | 의미 |
|---|---|
| `generation_latency_ms` | LLM 생성 시간 (t4 − t3) |
| `total_generation_module_latency_ms` | 생성 모듈 전체 시간 (t5 − t0) |
| `input_token_count` | LLM 입력 토큰 수 |
| `output_token_count` | LLM 출력 토큰 수 |
| `tokens_per_second` | 초당 출력 토큰 (gen latency 0이면 null) |
| `format_compliance` | 구분자/형식 준수 + 집합 일치 + `cited⊆used` 모두 통과 여부 |
| `inline_cited_chunk_ids` | 본문 인라인 `[Cxxx]`에서 추출한 ID(중복 허용) |
| `evidence_block_chunk_ids` | `<<<EVIDENCE>>>` 블록의 ID |
| `inline_evidence_set_match` | 인라인 집합 == Evidence 집합 일치 여부 |
| `content_missing_chunk_count` | content 없어 컨텍스트에서 제외된 chunk 수 |

> `is_warmup=True` 행은 cold start이므로 **latency 집계에서 제외**한다.

---

## 7. 문제 발생 시 확인할 것

| 증상 | 확인 |
|---|---|
| **VRAM 부족** (CUDA OOM) | `model.quantization` 을 `8bit`/`4bit`로. GPU를 A100/L4로 변경 |
| **bitsandbytes 설치 오류** | `!pip install -U bitsandbytes` 재설치, 런타임 재시작. 4/8bit 미사용 시 `none`으로 우회 |
| **Hugging Face 모델 다운로드 문제** | `login()` 으로 인증, 네트워크/rate limit 확인, 모델명 `Qwen/Qwen2.5-7B-Instruct` 철자 확인 |
| **Qwen 출력 형식 이탈** | `format_compliance=false` / `error_type` 확인. 빈도 높으면 프롬프트·few-shot 보정 필요(다음 단계) |
| **quantization 설정** | config의 `model.quantization` 값과 실제 VRAM이 맞는지 확인 |
| **경로 오류** | `%cd recrag_generation` 위치에서 실행했는지 확인. config는 상대경로를 자동으로 절대경로화함 |

---

## 8. GitHub 업로드 명령어 (Warp 터미널)

처음 올릴 때:
```bash
cd recrag_generation

git init
git add .
git commit -m "Initial RECRAG generation module raw baseline"

git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

이미 git repo가 초기화되어 있으면:
```bash
git add .
git commit -m "Update RECRAG generation module"
git push
```

---

## 9. Colab 실행 전 자체 점검 체크리스트

```text
[ ] requirements.txt 존재
[ ] generation_experiment_config.yaml 존재
[ ] data/eval_set.sample.jsonl 존재
[ ] src/run_generation.py 존재
[ ] outputs/ 폴더는 없어도 실행 시 생성 가능
[ ] .gitignore에 outputs/ 포함
[ ] Qwen2.5-7B-Instruct 모델명 확인 (Qwen/Qwen2.5-7B-Instruct)
[ ] quantization 설정 확인 (none / 4bit / 8bit)
[ ] Colab GPU 확인 (!nvidia-smi)
[ ] Hugging Face 로그인 필요 여부 확인
```

---

## 10. 프로젝트 구조

```
recrag_generation/
├── README.md
├── colab_run_commands.md
├── .gitignore
├── requirements.txt
├── generation_experiment_config.yaml   # 단일 진실 소스 (모든 파라미터)
├── data/
│   └── eval_set.sample.jsonl           # 샘플 4건
├── src/
│   ├── config_loader.py                # config 로드·검증
│   ├── data_loader.py                  # 평가셋 로드 + content 필터링 (embedding 제외)
│   ├── prompt_builder.py               # Raw Context 프롬프트(ChatML) 생성
│   ├── llm_runner.py                   # Qwen2.5-7B 추론, 토큰 카운트
│   ├── output_parser.py                # <<<...>>> 파싱, 인라인/Evidence 인용 추출
│   ├── latency_logger.py               # t0~t5 latency 측정, JSONL/CSV 저장
│   └── run_generation.py               # 실행 엔트리포인트
└── outputs/                            # (gitignore) 실행 시 자동 생성
    ├── generation_results.jsonl
    └── generation_results.csv
```
