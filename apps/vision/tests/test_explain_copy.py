from cropmerge.anomaly.explain import combined_review_score, explain_cell, recommendation
from cropmerge.anomaly.grid import GridCell


def test_explain_cell_uses_real_numbers():
    cell = GridCell(0, 0, 0, 10, 0, 10, 0.9, True)
    cell.features = {
        "coverage_delta": -0.18,
        "crop_coverage": 0.42,
        "bare_soil": 0.21,
        "color_difference": 1.35,
        "vegetation_difference": 0.08,
        "exg_difference": 0.08,
        "texture_difference": 0.55,
        "embedding_difference": 0.31,
    }
    cell.contributions = {
        "coverage": 0.3,
        "color": 0.15,
        "vegetation": 0.12,
        "texture": 0.1,
        "dino": 0.09,
        "isolation": 0.08,
    }
    reasons = explain_cell(cell, {"anomaly": {"reason_contribution_min": 0.08}}, persistence=0.72)
    assert any("−18%" in r or "-18%" in r for r in reasons)
    assert any("Lab distance 1.35" in r for r in reasons)
    assert any("72%" in r for r in reasons)


def test_review_score_formula():
    assert round(combined_review_score(0.8, 0.6), 2) == round(0.65 * 0.8 + 0.35 * 0.6, 2)


def test_recommendation_mentions_review_score():
    text = recommendation(
        "high",
        location="northwest",
        anomaly_score=0.82,
        persistence=0.71,
        frames_seen=8,
    )
    assert "review score" in text.lower()
    assert "northwest" in text
    assert "8 observations" in text
