"""Production ASGI wrapper with readiness, metrics, and proxy identity checks."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from compute_executor import get_compute_executor
from compute_governor import get_compute_governor
from observability import (
    configure_open_telemetry,
    prometheus_payload,
    shutdown_open_telemetry,
)
from resource_policy import ResourcePolicy

PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCE_POLICY = ResourcePolicy.from_environment()
PUBLIC_PROBE_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def _truthy_environment(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _server_port() -> int:
    try:
        port = int(os.getenv("DBF_SERVER_PORT", "8501"))
    except ValueError as error:
        raise ValueError("DBF_SERVER_PORT must be an integer.") from error
    if not 1 <= port <= 65_535:
        raise ValueError("DBF_SERVER_PORT must be between 1 and 65535.")
    return port


async def health(request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def readiness(request) -> JSONResponse:
    governor = get_compute_governor(RESOURCE_POLICY)
    executor = get_compute_executor(RESOURCE_POLICY)
    ready = governor.ready()
    return JSONResponse(
        {
            "status": "ready" if ready else "not-ready",
            "coordination": governor.snapshot().coordination_backend,
            "compute_backend": executor.snapshot().mode,
        },
        status_code=200 if ready else 503,
    )


async def metrics(request) -> Response:
    payload = prometheus_payload()
    if payload is None:
        return Response(
            "Prometheus support is not installed. Sync the ops extra.\n",
            status_code=503,
            media_type="text/plain",
        )
    body, content_type = payload
    return Response(body, media_type=None, headers={"Content-Type": content_type})


class ProxyIdentityMiddleware:
    """Require an identity header set by the private oauth2-proxy/Nginx tier."""

    def __init__(self, app) -> None:
        self.app = app
        self.required = _truthy_environment("DBF_REQUIRE_PROXY_IDENTITY", False)

    async def __call__(self, scope, receive, send) -> None:
        if (
            not self.required
            or scope.get("path") in PUBLIC_PROBE_PATHS
            or scope.get("type") not in {"http", "websocket"}
        ):
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        identity = headers.get("x-auth-request-email", "").strip()
        if identity:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        response = JSONResponse(
            {"error": "OIDC authentication is required."},
            status_code=401,
        )
        await response(scope, receive, send)


class SecurityHeadersMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        async def send_with_headers(message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", ()))
                headers.extend(
                    (
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (b"x-frame-options", b"SAMEORIGIN"),
                    )
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


@asynccontextmanager
async def lifespan(app):
    configure_open_telemetry()
    get_compute_governor(RESOURCE_POLICY)
    get_compute_executor(RESOURCE_POLICY)
    yield {"ready": True}
    shutdown_open_telemetry()


app = st.App(
    str(PROJECT_ROOT / "main.py"),
    routes=[
        Route("/healthz", health),
        Route("/readyz", readiness),
        Route("/metrics", metrics),
    ],
    middleware=[
        Middleware(SecurityHeadersMiddleware),
        Middleware(ProxyIdentityMiddleware),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    app.run(
        config={
            "server.address": os.getenv("DBF_SERVER_ADDRESS", "127.0.0.1"),
            "server.headless": True,
            "server.port": _server_port(),
        }
    )
