"""llm_runner.py — Qwen2.5-7B-Instruct 추론 (M5 LLM Runner).

책임:
- 모델/토크나이저 로드 (Colab GPU, 필요 시 4bit/8bit quantization)
- config 디코딩 파라미터 주입 (greedy decoding 기준)
- ChatML messages -> 텍스트 생성
- input/output token count 정확 기록

비스트리밍(일괄 생성). t4(생성 완료)는 호출자(run_generation)에서 측정한다.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg["model"]
        self.model_id = m["hf_model_id"]
        self.temperature = float(m["temperature"])
        self.top_p = float(m["top_p"])
        self.max_output_tokens = int(m["max_output_tokens"])
        self.repetition_penalty = m.get("repetition_penalty", None)
        self.quantization = (m.get("quantization") or "none").lower()

        self.tokenizer = None
        self.model = None
        # greedy(temperature=0.0) 미지원 환경 대비 — 실제 적용 디코딩을 기록
        self.decoding_note = None

    # ── 로드 ────────────────────────────────────────────────
    def load(self):
        """모델/토크나이저 로드. quantization 설정에 따라 분기."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}

        if self.quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            if self.quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:  # 8bit
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        return self

    # ── 생성 ────────────────────────────────────────────────
    def _gen_kwargs(self) -> dict:
        """디코딩 파라미터 구성. temperature=0.0 -> greedy(do_sample=False)."""
        kwargs = {
            "max_new_tokens": self.max_output_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.temperature == 0.0:
            # greedy decoding. top_p/temperature 는 sampling 파라미터이므로 전달하지 않는다.
            kwargs["do_sample"] = False
            self.decoding_note = "greedy(do_sample=False)"
        else:
            # [확인 필요] greedy 미사용 환경/요청 시 sampling. 로그에 명시된다.
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            self.decoding_note = f"sampling(temperature={self.temperature}, top_p={self.top_p})"

        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(self.repetition_penalty)
        return kwargs

    @torch.no_grad()
    def generate(self, messages: list[dict]) -> dict:
        """messages -> 생성. 출력 텍스트와 토큰 수를 반환한다.

        반환: {output_text, input_token_count, output_token_count, decoding_note}
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMRunner.load() 를 먼저 호출해야 합니다.")

        # 1) ChatML 템플릿을 '문자열'로 적용 (Qwen 공식 패턴)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) 토크나이즈 후 모델 디바이스로 이동
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_token_count = int(model_inputs["input_ids"].shape[-1])

        # 3) 생성
        output_ids = self.model.generate(**model_inputs, **self._gen_kwargs())

        # 4) 입력 길이 이후의 새 토큰만 출력으로 분리
        gen_ids = output_ids[0][input_token_count:]
        output_token_count = int(gen_ids.shape[-1])
        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return {
            "output_text": output_text,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "decoding_note": self.decoding_note,
        }
