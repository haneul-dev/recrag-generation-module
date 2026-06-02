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

## 참고
- `is_warmup=True` 행은 cold start이므로 **latency 집계에서 제외**한다.
- `outputs/` 는 `.gitignore` 대상이라 repo에 올라가지 않는다. 실행하면 자동 생성된다.
