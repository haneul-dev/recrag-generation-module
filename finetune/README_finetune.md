# RECRAG 생성 모듈 — QLoRA 파인튜닝 + 정량 평가 (논문급)

Qwen2.5-7B-Instruct를 **고정 추론**하던 생성 모듈에, **파인튜닝(QLoRA)** 와
**누수 없는 정량 평가**를 추가한 모듈이다.
목표: **(1) 출력형식·인용·근거성 준수**, **(2) 답변 품질** 동시 개선.
환경은 기존 실험과 동일한 **Colab T4(4bit)**.

> 핵심: 새 LLM을 만드는 게 아니라 base는 4bit로 얼리고 **LoRA 어댑터만 학습**한다.
> 산출물(= "파인튜닝한 모델")은 `finetune/outputs/.../` 의 LoRA 어댑터다.

## 구성

| 파일 | 역할 |
|---|---|
| `finetune_config.yaml` | 단일 설정 소스 (teacher / 분할 / 증강 / 품질 / LoRA / 학습 / 평가) |
| `build_sft_dataset.py` | 합성 학습셋 생성 — **누수 없는 분할** + 신규쿼리 합성 + 품질필터 + manifest |
| `train_qlora.py` | QLoRA SFT 학습 → LoRA 어댑터 저장 |
| `evaluate.py` | **held-out test 정량 평가**(repeats 반복측정·zero/few-shot) + 비교표 |
| `build_synth_test.py` | 합성 held-out 평가셋 확대(약지도, 누수 가드) |
| `merge_adapter.py` | LoRA 어댑터를 base에 병합(추론 latency 공정화/배포) |
| `colab_finetune.ipynb` | 원클릭 Colab 런북 (clone→학습→평가→merge) |

## 논문급 설계

**누수 차단**
- base **쿼리 단위**로 train/val/test 분할(예시 단위 아님 → 패러프레이즈 누수 방지).
- test = 학습에 안 쓴 **held-out 원본 쿼리**.
- abstain(X) 쿼리는 **계층 분할**로 test/val에도 커버리지 보장.
- 코퍼스 기반 **신규 합성 쿼리**는 train 한정 + val/test와 토큰 자카드 높으면 폐기.

**데이터 규모·품질**
- 패러프레이즈(train) + 코퍼스 기반 신규 쿼리 합성으로 확대.
- 형식·인용 정규화 검증(`output_parser`) + 길이/중복 필터.
- `dataset_manifest.json`: 시드·teacher 모델·분할 query_id·SHA256·누수통제 기록(재현성).

**정량 지표** (`evaluate.py`)
- 답가능: `answer_token_f1`, `answer_char_f1`, citation P/R/F1, `citation_exact_rate`, `format_compliance`
- 답불가: `abstain_ok_rate`, `hallucination_rate`, `format_compliance`
- 공통: latency p50/p95/mean

## 실행 (Colab 원클릭)

`colab_finetune.ipynb` 를 열어 위에서부터 실행. 또는 CLI:

```bash
pip install -r requirements.txt && pip install -r finetune/requirements_finetune.txt
export UPSTAGE_API_KEY="..."   # teacher (또는 OPENAI_API_KEY)

# 1) (선택) 데이터셋 재생성 — repo에 이미 포함됨
python finetune/build_sft_dataset.py --config finetune/finetune_config.yaml

# 2) QLoRA 학습 (GPU 필요)
python finetune/train_qlora.py --config finetune/finetune_config.yaml

# 3) 정량 평가 (메모리 안전: 각각 별도 프로세스)
python finetune/evaluate.py --config finetune/finetune_config.yaml --tag base
python finetune/evaluate.py --config finetune/finetune_config.yaml --tag finetuned \
    --adapter finetune/outputs/qwen2.5-7b-recrag-qlora
python finetune/evaluate.py --compare \
    finetune/outputs/eval/base.summary.json finetune/outputs/eval/finetuned.summary.json
```

## 한계 (논문에 솔직히 명시할 것)

자동화로 **여기까지** 했고, 아래는 **사람/추가 데이터가 필요**하다.

- **데이터 규모**: base 쿼리 30개 기반(+합성). teacher 합성은 **약지도(weak supervision)** 이며
  gold가 아니다 → 일부를 **사람이 검수**해 신뢰도를 보고해야 한다.
- **멀티모달**: 현재 evidence는 텍스트화본 위주. 실제 image/audio 데이터는 별도 구축 필요.
- **코퍼스 규모**: 200청크(하한). 본 프로젝트 목표(10K/100K/500K)와는 별개 실험.
- **포지셔닝**: 본 연구의 메인 메시지는 "생성 LLM 고정 + 검색 비교". 파인튜닝은
  **ablation/확장**으로 다루는 게 일관적이다.
- 평가 지표 answer_f1은 표면 유사도 → 의미 평가(사람/LLM-as-Judge)는 확장.
- 어댑터는 base 모델과 **동일 버전**으로 로드한다(`base_model` 고정).
