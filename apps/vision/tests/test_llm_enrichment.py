from api.main import _enrich_zones_with_llm
from cropmerge.enrich.llm_explain import is_loaded, llm_enrichment_enabled, summarize_zone, unload


def test_unload_is_safe_when_nothing_loaded():
    assert is_loaded() is False
    unload()  # must not raise
    assert is_loaded() is False


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CROP_MERGE_LLM_ENRICHMENT", raising=False)
    assert llm_enrichment_enabled() is False


def test_summarize_zone_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("CROP_MERGE_LLM_ENRICHMENT", "false")
    assert summarize_zone(primary_signal_label="x", reasons=[], location="center", review_priority="low") is None


def test_summarize_zone_returns_none_when_enabled_but_no_model_path(monkeypatch):
    monkeypatch.setenv("CROP_MERGE_LLM_ENRICHMENT", "true")
    monkeypatch.delenv("CROP_MERGE_LLM_MODEL_PATH", raising=False)
    assert summarize_zone(primary_signal_label="x", reasons=[], location="center", review_priority="low") is None


def test_enrich_zones_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("CROP_MERGE_LLM_ENRICHMENT", "false")
    report = {"inspectionZones": [{"id": "z1", "reasons": ["r1"]}]}
    _enrich_zones_with_llm(report)
    assert "llmSummary" not in report["inspectionZones"][0]


def test_enrich_zones_never_raises_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("CROP_MERGE_LLM_ENRICHMENT", "true")
    monkeypatch.setenv("CROP_MERGE_LLM_MODEL_PATH", "/nonexistent/path/does/not/exist")
    report = {"inspectionZones": [{"id": "z1", "reasons": ["r1"], "primarySignalLabel": "x", "relativeLocation": "center", "reviewPriority": "low"}]}
    _enrich_zones_with_llm(report)  # must not raise
    assert "llmSummary" not in report["inspectionZones"][0]
