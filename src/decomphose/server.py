from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from decomphose import __version__
from decomphose.clients.openrouter import OpenRouterClient
from decomphose.config import get_settings
from decomphose.middleware.harness import parse_harness_strategy
from decomphose.strategies import dispatch_strategy
from decomphose.strategies.context import StrategyContext
from decomphose.utils.errors import HarnessError
from decomphose.utils.logging import log_with_meta

log = logging.getLogger("decomphose.server")


def create_app(client: OpenRouterClient | None = None) -> FastAPI:
    settings = get_settings()
    upstream = client or OpenRouterClient(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await upstream.close()

    app = FastAPI(title="Decomphose", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "decomphose", "version": __version__}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            meta = parse_harness_strategy(request)
            body: dict[str, Any] = await request.json()

            log_with_meta(
                log,
                logging.INFO,
                "Incoming chat completion",
                {
                    "requestId": meta.request_id,
                    "strategy": meta.strategy.value,
                    "clientModel": body.get("model"),
                    "stream": bool(body.get("stream")),
                },
            )

            result = await dispatch_strategy(
                StrategyContext(
                    settings=settings,
                    client=upstream,
                    meta=meta,
                    body=body,
                )
            )

            response_headers = {
                **result.headers,
                "x-harness-request-id": meta.request_id,
            }

            if result.stream is not None:
                return StreamingResponse(
                    result.stream,
                    status_code=result.status,
                    media_type="text/event-stream",
                    headers={
                        **response_headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            return JSONResponse(
                content=result.body,
                status_code=result.status,
                headers=response_headers,
            )
        except HarnessError as exc:
            log_with_meta(log, logging.WARNING, "Harness error", {"code": exc.code, "message": exc.message})
            return JSONResponse(
                status_code=exc.status,
                content={"error": {"message": exc.message, "type": exc.code, "code": exc.code}},
            )
        except Exception as exc:
            log_with_meta(log, logging.ERROR, "Unhandled error", {"message": str(exc)})
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "Internal harness error", "type": "INTERNAL_ERROR"}},
            )

    return app


def run() -> None:
    import uvicorn

    settings = get_settings()
    log_with_meta(
        log,
        logging.INFO,
        "Decomphose proxy listening",
        {
            "url": f"http://{settings.harness_host}:{settings.harness_port}",
            "health": "/health",
            "completions": "/v1/chat/completions",
        },
    )
    uvicorn.run(
        "decomphose.server:create_app",
        factory=True,
        host=settings.harness_host,
        port=settings.harness_port,
        log_level="info",
    )
