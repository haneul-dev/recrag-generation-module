# RECRAG 생성 모듈 — QLoRA 파인튜닝

Qwen2.5-7B-Instruct를 **고정 추론**하던 생성 모듈에, **파인튜닝(QLoRA)** 단계를 추가한 모듈이다.
목표는 두 가지: **(1) 출력형식·인용·근거성 준수**, **(2) 답변 품질(answer_f1)** 동시 개선.
환경은 기존 실험과 동일한 **Colab T4(4bit)** 기준.

> 핵심: 새 LLM을 만드는 게 아니라, base 모델은 4bit로 얼리고 **LoRA 어댑터만 학습**한다.
> 산출물(= "파인튜닝한 모델")은 `finetune/outputs/.../` 의 LoRA 어댑터다.

## 구성

| 파일 | 역할 |
|---|---|
| `finetune_config.yaml` | 단일 설정 소스 (teacher / 증강 / LoRA / 학습 하이퍼파라미터) |
| `build_sft_dataset.py` | teacher LLM으로 합성 학습셋 생성 + 형식검증 → `data/sft_{train,val}.jsonl` |
| `train_qlora.py` | QLoRA SFT 학습 → LoRA 어댑터 저장 |
| `requirements_finetune.txt` | 추가 의존성 |

학습 데이터의 **정답 타깃은 teacher LLM(합성)으로 생성**하며, `output_parser`로 형식·인용을
검증해 통과한 것만 학습에 넣는다. 추론과 동일한 `prompt_builder`/`output_parser`를 재사용한다.

## 실행 순서 (Colab)

```bash
# 0) 의존성
pip install -r requirements.txt
pip install -r finetune/requirements_finetune.txt

# 1) teacher API 키 (Upstage Solar 예시; OpenAI면 OPENAI_API_KEY)
export UPSTAGE_API_KEY="..."

# 2) 합성 학습셋 생성 (경로/개수만 먼저 점검하려면 --dry-run)
python finetune/build_sft_dataset.py --config finetune/finetune_config.yaml --dry-run
python finetune/build_sft_dataset.py --config finetune/finetune_config.yaml

# 3) QLoRA 학습 (GPU 런타임 필요)
python finetune/train_qlora.py --config finetune/finetune_config.yaml
# 결과: finetune/outputs/qwen2.5-7b-recrag-qlora/  (LoRA 어댑터 + manifest)
```

## 파인튜닝 모델로 추론/평가

기존 추론 코드를 그대로 쓰되, config의 `model.adapter_path`만 어댑터 경로로 지정하면 된다
(`llm_runner`가 base 위에 어댑터를 얹어 로드).

```yaml
# generation_experiment_config.yaml
model:
  hf_model_id: "Qwen/Qwen2.5-7B-Instruct"
  quantization: "4bit"
  adapter_path: "finetune/outputs/qwen2.5-7b-recrag-qlora"   # ← 추가
```

이후 `src/run_generation.py`로 baseline(어댑터 없음) vs 파인튜닝(어댑터 있음)을
같은 평가셋에서 비교하면 된다. **공정 비교를 위해 base와 동일한 디코딩/프롬프트/평가셋을 유지**한다.

## 주의 (과장 금지)

- 학습셋이 작으면(쿼리 30 기반) 과적합 위험 → 패러프레이즈/abstain 증강과 val loss를 함께 본다.
- teacher 합성 정답은 gold가 아니라 **약지도(weak supervision)** 다. 평가 수치는 실측으로만 보고한다.
- 어댑터는 base 모델과 **반드시 동일 버전**으로 로드한다(`base_model` 고정).
