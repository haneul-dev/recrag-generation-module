# Colab 실행 명령어 — RECRAG Step 8~9 생성 모듈 (Raw Baseline)

> Google Colab(paid GPU)에서 GitHub repo를 clone하여 실행하는 전체 흐름.
> repo: `https://github.com/haneul-dev/recrag-generation-module.git`
> clone하면 폴더명은 **`recrag-generation-module`** 이 된다(repo 이름과 동일).

---

## 0. 런타임 설정
- 상단 메뉴 → **런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU**
- paid plan 권장 GPU: A100 / L4 / (최소) T4

---

## 1. GPU 확인
```bash
!nvidia-smi
```
> 출력의 **GPU 종류와 VRAM**을 기록한다. VRAM이 부족하면 4단계에서 quantization을 설정한다.

---

## 2. GitHub repo clone
```bash
!git clone https://github.com/haneul-dev/recrag-generation-module.git
%cd /content/recrag-generation-module
```
> clone 폴더명은 repo 이름과 같은 `recrag-generation-module` 이다.

---

## 3. 패키지 설치
```bash
!pip install -r requirements.txt
```
> torch는 Colab 기본 설치본을 사용한다. 설치 후 런타임 재시작 안내가 뜨면 재시작 후 다시 `%cd /content/recrag-generation-module`.

---

## 4. (선택) VRAM 부족 시 quantization 설정
`generation_experiment_config.yaml` 의 `model.quantization` 을 수정한다.
```yaml
model:
  quantization: "4bit"   # none | 4bit | 8bit
```
- 24GB+ VRAM : `none`
- 16GB 내외 : `8bit` 또는 `4bit`

Colab 셀에서 바로 수정하려면:
```python
import yaml
p = "generation_experiment_config.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg["model"]["quantization"] = "4bit"   # 필요 시
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
```

---

## 5. (필요 시) Hugging Face 로그인
Qwen2.5-7B-Instruct는 공개 모델이라 보통 로그인 없이 다운로드되지만,
다운로드가 막히거나 rate limit이 걸리면 로그인한다.
```python
from huggingface_hub import login
login()   # 실행하면 토큰 입력창이 뜬다. 거기에 붙여넣는다.
```
> ⚠️ **토큰을 코드/노트북 셀에 직접 적지 말 것.** `login()` 의 입력창에만 붙여넣고,
> repo·config·소스 어디에도 토큰을 저장하지 않는다.

---

## 6. 실행
```bash
!python src/run_generation.py --config generation_experiment_config.yaml
```

---

## 7. 결과 확인
```bash
!ls outputs
!head -n 5 outputs/generation_results.csv
```
```python
import pandas as pd
df = pd.read_csv("outputs/generation_results.csv")
df[["query_id","run_id","is_warmup","generation_latency_ms",
    "input_token_count","output_token_count","tokens_per_second",
    "format_compliance","inline_evidence_set_match","groundedness_note"]]
```

---

## 8. (본 실행 전) 검증 케이스 2건

본 실험 전, 별도 산출물(`--run-tag verify`)로 error 경로와 model-driven abstain을 확인한다.
검증용 평가셋 `data/eval_set.verify.jsonl` 을 사용한다(V001=무관 content, V002=정상).

### 8-1. error 경로 강제 테스트 (status=error row 기록 확인)
```bash
!python src/run_generation.py --config generation_experiment_config.yaml \
    --eval-set data/eval_set.verify.jsonl --force-error-on V002 --run-tag verify_error
```
```python
import pandas as pd
df = pd.read_csv("outputs/generation_results.verify_error.csv")
print(df[df.query_id=="V002"][["query_id","run_id","status","row_kind","error_type","error_message"]].head())
# 기대: V002 의 row_kind=llm_error, status=error, error_message 채워짐
```

### 8-2. model-driven abstain 테스트 (input_token_count>0 인데 모델이 "근거 부족")
```bash
!python src/run_generation.py --config generation_experiment_config.yaml \
    --eval-set data/eval_set.verify.jsonl --run-tag verify_abstain
```
```python
import pandas as pd
df = pd.read_csv("outputs/generation_results.verify_abstain.csv")
print(df[(df.query_id=="V001") & (~df.is_warmup)][
    ["query_id","row_kind","llm_invoked","input_token_count","groundedness_note","cited_chunk_ids"]].head())
# 기대: V001 row_kind=llm_success, llm_invoked=True, input_token_count>0,
#        groundedness_note="근거 부족"(모델이 추측 없이 기권)
```

> 검증 산출물은 `--run-tag` 덕분에 `*.verify_error.*` / `*.verify_abstain.*` 로 분리 저장되어
> 본 실행 결과(`generation_results.jsonl/csv`)를 덮어쓰지 않는다.

---

## 9. eval set 검증 + interim 예비 실행

### 9-1. 본 실행 전 eval set 검증 (스키마/라벨 품질)
두 방법 중 하나. FAIL 이면 종료코드 1.
```bash
# A안: run_generation 의 --validate-only (모델 로드 안 함)
!python src/run_generation.py --config generation_experiment_config.yaml \
    --eval-set data/eval_set.interim.jsonl --validate-only

# B안: standalone 스크립트
!python src/validate_eval_set.py --eval-set data/eval_set.interim.jsonl
```
> 결과가 `FAIL` 이면 출력된 `✗` 항목을 고친 뒤 실행한다. `WARN` 은 interim 예비에서는 대개 허용.

### 9-2. interim 본 실행
```bash
!python src/run_generation.py \
  --config generation_experiment_config.yaml \
  --eval-set data/eval_set.interim.jsonl \
  --run-tag rawbase_interim_4bit_20260604 \
  --experiment-id EXP-GEN-RAWBASE-INTERIM-001
```
생성 산출물(`--run-tag` 로 분리):
- `outputs/generation_results.rawbase_interim_4bit_20260604.jsonl`
- `outputs/generation_results.rawbase_interim_4bit_20260604.csv`
- `outputs/run_metadata.rawbase_interim_4bit_20260604.json`

> ⚠️ interim 데이터는 정식 검색 스냅샷이 아니다. `--run-tag`(rawbase_**interim**_…)와
> `--experiment-id`(EXP-GEN-RAWBASE-**INTERIM**-001)로 정식 실행과 반드시 구분한다.
> 이 두 옵션은 config 기본값을 건드리지 않고 이번 실행에만 적용된다(정식 실행용 config 보존).

---

## 참고
- `is_warmup=True` 행은 cold start이므로 **latency 집계에서 제외**한다.
- **deterministic abstain**(content 전무 → LLM 미호출, `row_kind=deterministic_abstain`) 행도
  **LLM latency 집계에서 제외**된다. latency 통계는 `row_kind=llm_success` & 측정 row 만 사용한다.
- 매 실행마다 `outputs/run_metadata.json` 에 **quantization / device / peak_vram_gb / latency 요약**이 기록된다.
- `outputs/` 는 `.gitignore` 대상이라 repo에 올라가지 않는다. 실행하면 자동 생성된다.
