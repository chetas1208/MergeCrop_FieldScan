from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("cropmerge")
    if root.handlers:
        root.setLevel(level.upper())
        return root
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    return root


def setup_access_logging(
    *,
    log_path: str | None = None,
    level: str = "INFO",
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Structured JSON access log for browser / Cloudflare traffic."""
    logger = logging.getLogger("cropmerge.api.access")
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_path:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
