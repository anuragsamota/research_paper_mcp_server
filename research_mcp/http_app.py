"""Streamable-HTTP app with optional bearer-token auth (RESEARCH_API_TOKEN)."""

from __future__ import annotations

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.transport_security import TransportSecuritySettings

from .server import mcp


class TokenAuth(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return JSONResponse({"status": "ok"})
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if self.token and not hmac.compare_digest(supplied, self.token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def create_app():
    token = os.environ.get("RESEARCH_API_TOKEN", "")
    if token:
        # With a token, Host-header (DNS-rebinding) checks are relaxed so the server works over the
        # LAN, in Docker and behind reverse proxies. Without one, only localhost Host headers pass.
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = mcp.streamable_http_app()
    app.add_middleware(TokenAuth, token=token)
    return app
