from __future__ import annotations

import json
import logging
from pathlib import Path

from decomphose.types import FrontierModelsConfig
from decomphose.utils.errors import HarnessError
from decomphose.utils.logging import log_with_meta

log = logging.getLogger("decomphose.registry")


class ModelRegistry:
    """Frontier model registry with hot reload.

    Each load() stats the backing file; if it changed since the last parse, the
    registry re-reads it. An invalid edit never takes down a running proxy —
    the last good config keeps being served and a warning is logged.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cached: FrontierModelsConfig | None = None
        self._stamp: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> FrontierModelsConfig:
        try:
            stat = self._path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            return self._fallback_or_raise(f"Registry file unreadable: {exc}", exc)

        if self._cached is not None and stamp == self._stamp:
            return self._cached

        try:
            raw = self._path.read_text(encoding="utf-8")
            config = FrontierModelsConfig.model_validate(json.loads(raw))
        except Exception as exc:  # noqa: BLE001 — any bad edit must not crash the proxy.
            return self._fallback_or_raise(f"Registry file invalid: {exc}", exc)

        if not config.models:
            return self._fallback_or_raise("Registry must contain at least one model", None)

        if self._cached is not None:
            log_with_meta(
                log,
                logging.INFO,
                "Model registry hot-reloaded",
                {
                    "path": str(self._path),
                    "version": config.version,
                    "modelCount": len(config.models),
                },
            )

        self._cached = config
        self._stamp = stamp
        return config

    def _fallback_or_raise(
        self, message: str, cause: BaseException | None
    ) -> FrontierModelsConfig:
        if self._cached is not None:
            log_with_meta(
                log,
                logging.WARNING,
                "Registry reload failed — serving last good config",
                {"path": str(self._path), "error": message},
            )
            return self._cached
        raise HarnessError(message, "INVALID_REGISTRY", 500, cause)
