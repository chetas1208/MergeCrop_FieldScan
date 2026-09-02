"""Anomaly package — lazy exports."""

__all__ = ["score_frame", "aggregate_zones"]


def __getattr__(name: str):
    if name == "score_frame":
        from cropmerge.anomaly.spatial import score_frame

        return score_frame
    if name == "aggregate_zones":
        from cropmerge.anomaly.temporal import aggregate_zones

        return aggregate_zones
    raise AttributeError(name)
