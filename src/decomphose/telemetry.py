from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

from decomphose import __version__
from decomphose.utils.logging import log_with_meta

if TYPE_CHECKING:
    from decomphose.config import HarnessSettings

log = logging.getLogger("decomphose.telemetry")

_provider: TracerProvider | None = None


def get_tracer() -> trace.Tracer:
    """Tracer for harness spans — a no-op unless configure_telemetry enabled the SDK."""
    return trace.get_tracer("decomphose", __version__)


def configure_telemetry(settings: HarnessSettings) -> None:
    """Install a TracerProvider when HARNESS_OTEL_ENABLED is set; otherwise stay no-op."""
    global _provider
    if _provider is not None or not settings.harness_otel_enabled:
        return

    resource = Resource.create(
        {"service.name": "decomphose", "service.version": __version__}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(settings)))
    trace.set_tracer_provider(provider)
    _provider = provider

    log_with_meta(
        log,
        logging.INFO,
        "OpenTelemetry tracing enabled",
        {"endpoint": settings.otel_exporter_otlp_endpoint or "console"},
    )


def shutdown_telemetry() -> None:
    """Flush pending spans on server shutdown."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def _build_exporter(settings: HarnessSettings) -> SpanExporter:
    if settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            return OTLPSpanExporter()
        except ImportError:
            log_with_meta(
                log,
                logging.WARNING,
                "OTLP endpoint set but exporter missing — install decomphose[otel]; "
                "falling back to console exporter",
                {"endpoint": settings.otel_exporter_otlp_endpoint},
            )
    return ConsoleSpanExporter()
