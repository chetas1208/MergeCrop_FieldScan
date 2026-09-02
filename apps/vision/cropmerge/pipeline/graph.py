"""Pipeline compute graph — staged DAG with timing, deps, artifacts."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("cropmerge.graph")


@dataclass
class StageResult:
    name: str
    ok: bool
    latency_sec: float
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Stage:
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    requires: tuple[str, ...] = ()
    optional: bool = False


class PipelineGraph:
    """
    Linear-ish DAG executor.

    Nodes write into a shared ctx. Downstream stages declare requires.
    """

    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self.history: list[StageResult] = []

    def run(self, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = dict(ctx or {})
        ctx.setdefault("_latency", {})
        for stage in self.stages:
            missing = [r for r in stage.requires if r not in ctx]
            if missing:
                msg = f"stage {stage.name} missing deps: {missing}"
                if stage.optional:
                    log.warning(msg)
                    self.history.append(StageResult(stage.name, False, 0.0, error=msg))
                    continue
                raise RuntimeError(msg)

            t0 = time.perf_counter()
            log.info("▶ %s", stage.name)
            try:
                out = stage.fn(ctx) or {}
                dt = time.perf_counter() - t0
                ctx.update(out)
                ctx["_latency"][stage.name] = round(dt, 4)
                self.history.append(StageResult(stage.name, True, dt, outputs=dict(out)))
                log.info("✔ %s (%.3fs)", stage.name, dt)
            except Exception as e:
                dt = time.perf_counter() - t0
                log.exception("✖ %s failed", stage.name)
                self.history.append(StageResult(stage.name, False, dt, error=str(e)))
                if stage.optional:
                    continue
                raise
        return ctx

    def mermaid(self) -> str:
        lines = ["flowchart TD"]
        prev = None
        for s in self.stages:
            nid = s.name.replace(".", "_").replace("-", "_")
            label = s.name
            if s.optional:
                label += " (opt)"
            lines.append(f'  {nid}["{label}"]')
            if prev:
                lines.append(f"  {prev} --> {nid}")
            prev = nid
        return "\n".join(lines)

    def latency_map(self) -> dict[str, float]:
        return {h.name: round(h.latency_sec, 4) for h in self.history}
