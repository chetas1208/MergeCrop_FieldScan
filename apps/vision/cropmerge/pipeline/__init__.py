"""Pipeline package. Import processor explicitly to avoid circular imports."""

__all__ = ["FieldTriageProcessor", "FieldTriageReport"]


def __getattr__(name: str):
    if name == "FieldTriageProcessor":
        from cropmerge.pipeline.processor import FieldTriageProcessor

        return FieldTriageProcessor
    if name == "FieldTriageReport":
        from cropmerge.pipeline.schemas import FieldTriageReport

        return FieldTriageReport
    raise AttributeError(name)
