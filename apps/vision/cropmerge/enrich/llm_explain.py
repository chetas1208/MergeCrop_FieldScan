from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("cropmerge.enrich.llm_explain")

ENABLED_ENV = "CROP_MERGE_LLM_ENRICHMENT"
MODEL_PATH_ENV = "CROP_MERGE_LLM_MODEL_PATH"
DEVICE_ENV = "CROP_MERGE_LLM_DEVICE"
MAX_NEW_TOKENS = 120

_lock = threading.Lock()
_model = None
_tokenizer = None
_load_failed = False

SYSTEM_PROMPT = (
    "You rewrite structured crop-field visual analysis observations into a short, "
    "plain-language note for a farmer reviewing drone field-scan results. "
    "Rules: 1-2 sentences only. Never diagnose disease, nutrient deficiency, irrigation "
    "failure, or plant health — only describe the visual observation. Never invent facts "
    "not present in the input. Plain, direct language."
)


def llm_enrichment_enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "false").strip().lower() == "true"


def _model_path() -> str | None:
    return os.environ.get(MODEL_PATH_ENV, "").strip() or None


def is_loaded() -> bool:
    """Cheap, non-blocking check for /vision/health — never triggers a load."""
    return _model is not None


def health_status() -> dict:
    return {
        "enabled": llm_enrichment_enabled(),
        "modelConfigured": bool(_model_path()),
        "loaded": is_loaded(),
        "device": _pick_device() if llm_enrichment_enabled() else None,
        "model": os.path.basename(_model_path()) if _model_path() else None,
    }


def _pick_device() -> str:
    configured = os.environ.get(DEVICE_ENV, "").strip()
    if configured:
        return configured
    try:
        import torch

        # Keep the enrichment model off the CV pipeline's GPU when a second
        # GPU is available, so the two don't contend for VRAM.
        return "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


def _ensure_loaded() -> bool:
    global _model, _tokenizer, _load_failed
    if _model is not None:
        return True
    if _load_failed or not llm_enrichment_enabled():
        return False
    path = _model_path()
    if not path:
        log.warning("%s is enabled but %s is not set — enrichment disabled", ENABLED_ENV, MODEL_PATH_ENV)
        _load_failed = True
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = _pick_device()
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
            model = model.to(device).eval()
            _tokenizer, _model = tokenizer, model
            log.info("LLM enrichment model loaded on %s from %s", device, path)
            return True
        except Exception as e:
            log.warning("LLM enrichment model failed to load (%s) — disabling for this process", e)
            _load_failed = True
            return False


def summarize_zone(
    *,
    primary_signal_label: str,
    reasons: list[str],
    location: str,
    review_priority: str,
) -> str | None:
    """Return a short farmer-facing paragraph for one Inspection Area, or
    None if enrichment is disabled/unavailable/failed. Callers must fall
    back to the existing heuristic `reasons` text in that case — never
    show a placeholder in its place.
    """
    if not _ensure_loaded():
        return None
    try:
        import torch

        user_prompt = (
            f"Location in field: {location}\n"
            f"Review priority: {review_priority}\n"
            f"Signal: {primary_signal_label}\n"
            f"Evidence: {'; '.join(reasons) if reasons else 'none'}\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
        with torch.inference_mode():
            output = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        result = _tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result or None
    except Exception as e:
        log.warning("LLM enrichment generation failed (%s)", e)
        return None
