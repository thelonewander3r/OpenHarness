from __future__ import annotations

import json
import logging
from typing import Any


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"decomphose.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_with_meta(logger: logging.Logger, level: int, msg: str, meta: dict[str, Any] | None = None) -> None:
    suffix = f" {json.dumps(meta)}" if meta else ""
    logger.log(level, f"{msg}{suffix}")
