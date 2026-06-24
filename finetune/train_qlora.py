"""train_qlora.py — Qwen2.5-7B-Instruct QLoRA SFT (Colab T4).

특징:
- base 모델 4bit(nf4) 로드 + LoRA 어댑터만 학습 (QLoRA).
- completion-only 손실: 프롬프트(system/user) 토큰은 label=-100 으로 마스킹하고
  assistant 정답 토큰에 대해서만 loss 를 계산한다. (TRL 버전 의존 없이 직접 구현)
- 입력: build_sft_dataset.py 가 만든 {"messages":[...]} JSONL.
- 출력: LoRA 어댑터 (train.output_dir). 추론은 llm_runner 에 adapter_path 로 로드.

실행:
  python finetune/train_qlora.py --config finetune/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

# CUDA 메모리 단편화 완화 (torch import 전에 설정해야 적용됨)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _abspath(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


# ─────────────────────────────────────────────────────────────
# 토크나이즈: completion-only 마스킹
# ─────────────────────────────────────────────────────────────
def make_tokenize_fn(tokenizer, max_len: int):
    """{"messages":[...]} -> input_ids/labels. assistant 정답 외 토큰은 -100."""

    def _fn(example):
        messages = example["messages"]
        # chat 템플릿을 '문자열'로 먼저 받는다 (tokenize=True가 버전에 따라
        # tokenizers.Encoding 객체를 반환해 Arrow 직렬화가 깨지는 문제 회피).
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # 템플릿이 이미 특수토큰을 포함하므로 add_special_tokens=False.
        # 정수 리스트(list[int])로 보장.
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_len]
        labels = list(full_ids)
        # 프롬프트 구간 마스킹
        n_prompt = min(len(prompt_ids), len(full_ids))
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}

    return _fn


class PadCollator:
    """input_ids/labels 를 배치 최대 길이로 패딩하는 collator."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "finetune_config.yaml"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))

    t = cfg["train"]
    train_path = _abspath(repo_root, cfg["synth"]["output"]["train_path"])
    val_path = _abspath(repo_root, cfg["synth"]["output"]["val_path"])
    output_dir = _abspath(repo_root, t["output_dir"])

    # ── 토크나이저 ──
    tokenizer = AutoTokenizer.from_pretrained(t["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── base 모델 4bit 로드 ──
    q = t["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(q["load_in_4bit"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_DTYPE[q["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=bool(q["bnb_4bit_use_double_quant"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        t["base_model"], quantization_config=bnb, device_map="auto",
        torch_dtype=_DTYPE[q["bnb_4bit_compute_dtype"]],
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(t["gradient_checkpointing"])
    )

    # ── LoRA ──
    lora = t["lora"]
    model = get_peft_model(model, LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=lora["target_modules"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.config.use_cache = False  # gradient checkpointing 과 충돌 방지

    # ── 데이터셋 ──
    data_files = {"train": train_path}
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        data_files["validation"] = val_path
    ds = load_dataset("json", data_files=data_files)
    tok_fn = make_tokenize_fn(tokenizer, int(t["max_seq_len"]))
    ds = ds.map(tok_fn, remove_columns=ds["train"].column_names)
    has_val = "validation" in ds

    # ── 학습 인자 ──
    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 1)),
        eval_accumulation_steps=1,   # 평가 logits를 매 스텝 CPU로 내려 GPU 누적 방지
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t["num_train_epochs"]),
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=float(t["weight_decay"]),
        logging_steps=int(t["logging_steps"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t["optim"],
        fp16=True,
        seed=int(t["seed"]),
        report_to="none",
        eval_strategy="steps" if has_val else "no",
        eval_steps=int(t["eval_steps"]) if has_val else None,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_val else None,
        data_collator=PadCollator(tokenizer),
    )

    trainer.train()

    # ── 어댑터 저장 ──
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "recrag_finetune_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "base_model": t["base_model"],
            "lora": lora,
            "max_seq_len": t["max_seq_len"],
            "epochs": t["num_train_epochs"],
            "train_examples": len(ds["train"]),
        }, f, ensure_ascii=False, indent=2)
    print(f"[done] LoRA 어댑터 저장: {output_dir}")


if __name__ == "__main__":
    main()
