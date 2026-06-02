"""config_loader.py — generation_experiment_config.yaml 로드/검증.

설정 주도 원칙(Config-driven): 모든 파라미터는 yaml에서 읽고 하드코딩하지 않는다.
1차 Raw Baseline 실행에 필요한 최소 키만 검증한다.
"""

from __future__ import annotations

import os
import yaml


# 1차 실행에서 반드시 존재해야 하는 키 (섹션.키)
REQUIRED_KEYS = [
    "model.hf_model_id",
    "model.temperature",
    "model.top_p",
    "model.max_output_tokens",
    "prompt.context_type",
    "prompt.prompt_type",
    "prompt.delimiters",
    "context.top_k_input",
    "experiment.repeats",
    "experiment.warmup_runs",
    "data.eval_set_path",
    "logging.jsonl_path",
    "logging.csv_path",
]


def _get_nested(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def load_config(config_path: str) -> dict:
    """yaml 설정을 읽고 1차 실행 가정에 맞는지 검증해 dict로 반환한다."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config 최상위는 매핑(dict) 형태여야 합니다.")

    # 필수 키 검증
    missing = []
    for key in REQUIRED_KEYS:
        _, ok = _get_nested(cfg, key)
        if not ok:
            missing.append(key)
    if missing:
        raise ValueError(f"config 필수 키 누락: {missing}")

    # 1차 범위 가드: Raw Baseline 고정값 확인 (벗어나면 명시적으로 막는다)
    context_type = cfg["prompt"]["context_type"]
    if context_type != "raw":
        raise ValueError(
            f"1차 Raw Baseline은 context_type='raw'만 지원합니다 (현재: {context_type})."
        )

    # config 파일 위치를 기준으로 상대경로를 절대경로화 (Colab 작업경로 흔들림 방지)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    cfg["_base_dir"] = base_dir
    cfg["data"]["eval_set_path"] = _abspath(base_dir, cfg["data"]["eval_set_path"])
    cfg["logging"]["jsonl_path"] = _abspath(base_dir, cfg["logging"]["jsonl_path"])
    cfg["logging"]["csv_path"] = _abspath(base_dir, cfg["logging"]["csv_path"])

    return cfg


def _abspath(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def decoding_params_snapshot(cfg: dict) -> dict:
    """로그에 남길 디코딩 파라미터 스냅샷."""
    m = cfg["model"]
    return {
        "temperature": m["temperature"],
        "top_p": m["top_p"],
        "max_output_tokens": m["max_output_tokens"],
        "repetition_penalty": m.get("repetition_penalty"),
        "stop": m.get("stop_sequence"),
    }
