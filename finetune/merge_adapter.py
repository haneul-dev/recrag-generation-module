"""merge_adapter.py — LoRA 어댑터를 base에 병합해 단일 모델로 저장.

용도
----
- 추론 latency 공정화: 미병합 LoRA는 forward마다 어댑터 행렬을 더해 오버헤드가 있다.
  병합하면 base 가중치에 흡수되어 오버헤드가 사라진다.
- 배포: base+어댑터 2개 대신 단일 모델로 관리.

메모리 주의 (중요)
------------------
병합은 **fp16 전체 가중치**를 메모리에 올린다(7B ≈ 15GB). T4(VRAM 16GB)에서는
빠듯하므로 기본은 **CPU에서 병합**한다 → Colab **고용량 RAM 런타임(High-RAM)** 권장.
저장 결과(~15GB)는 Drive 등 영구 저장소로 옮길 것.

실행:
  python finetune/merge_adapter.py --config finetune/finetune_config.yaml \
      --adapter finetune/outputs/qwen2.5-7b-recrag-qlora \
      --out finetune/outputs/qwen2.5-7b-recrag-merged
병합 모델 추론: generation_experiment_config.yaml 의 model.hf_model_id 를 위 --out 경로로.
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "finetune_config.yaml"))
    ap.add_argument("--adapter", required=True, help="LoRA 어댑터 경로")
    ap.add_argument("--out", required=True, help="병합 모델 저장 경로")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="병합 디바이스(기본 cpu: 메모리 안전)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    base_model = cfg["train"]["base_model"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(repo_root, args.adapter)
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)

    print(f"[load] base(fp16) = {base_model} on {args.device}")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16,
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    print(f"[load] adapter = {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    print("[merge] merge_and_unload() ...")
    model = model.merge_and_unload()

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[done] 병합 모델 저장: {out}")


if __name__ == "__main__":
    main()
