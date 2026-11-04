"""HTTP service layer (FastAPI).

    POST /v1/documents            upsert a document into the retrieval index
    DELETE /v1/documents/{id}     remove a document
    POST /v1/ask                  grounded Q&A over the index (RAG pipeline)
    POST /v1/completions          raw gateway completion (routing, cache, failover)
    GET  /v1/traces/{trace_id}    span tree for a request
    GET  /v1/providers            circuit-breaker state and latency EWMA per provider
    GET  /healthz | /readyz       liveness / readiness
    GET  /metrics                 Prometheus exposition

Tenancy comes from the `X-Tenant-ID` header; domain errors map onto HTTP status
codes in one exception handler so routes stay thin.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from meridian import __version__
from meridian.core.config import EngineConfig
from meridian.core.errors import (
    BudgetExceededError,
    GuardrailViolation,
    MeridianError,
    NoHealthyProviderError,
    RateLimitedError,
)
from meridian.core.types import CompletionRequest, RoutingPolicy
from meridian.engine import Engine


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None
    session_id: str | None = None
    policy: RoutingPolicy = RoutingPolicy.QUALITY
    model: str | None = None


STATUS = {
    GuardrailViolation: 422,
    BudgetExceededError: 402,
    RateLimitedError: 429,
    NoHealthyProviderError: 503,
}


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or Engine(EngineConfig.load())
        yield
        await app.state.engine.aclose()

    app = FastAPI(title="Meridian Engine", version=__version__, lifespan=lifespan)

    def eng(request: Request) -> Engine:
        return request.app.state.engine

    @app.exception_handler(MeridianError)
    async def domain_error(_: Request, exc: MeridianError) -> JSONResponse:
        status = next((s for t, s in STATUS.items() if isinstance(exc, t)), 500)
        body: dict[str, Any] = {"error": type(exc).__name__, "detail": str(exc)}
        if isinstance(exc, GuardrailViolation):
            body["reasons"] = exc.reasons
        return JSONResponse(body, status_code=status)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        e = eng(request)
        providers = {name: e.gateway.breaker(name).state.value
                     for name in e.gateway._slots}
        ready = any(state != "open" for state in providers.values())
        if not ready:
            raise HTTPException(503, {"providers": providers})
        return {"ready": True, "providers": providers, "index": e.retrieval.stats}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request) -> str:
        return eng(request).metrics.render()

    @app.post("/v1/documents", status_code=201)
    async def upsert_document(doc: DocumentIn, request: Request) -> dict[str, Any]:
        n = await eng(request).ingest(doc.id, doc.text, doc.metadata)
        return {"id": doc.id, "chunks": n}

    @app.delete("/v1/documents/{doc_id:path}")
    async def delete_document(doc_id: str, request: Request) -> dict[str, Any]:
        n = eng(request).retrieval.delete(doc_id)
        if not n:
            raise HTTPException(404, f"document '{doc_id}' not found")
        return {"id": doc_id, "deleted_chunks": n}

    @app.post("/v1/ask")
    async def ask(body: AskIn, request: Request,
                  x_tenant_id: str = Header(default="default")) -> dict[str, Any]:
        res = await eng(request).ask(body.question, k=body.k, filters=body.filters,
                                     tenant=x_tenant_id, session_id=body.session_id,
                                     policy=body.policy, model=body.model)
        return res.__dict__

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request,
                          x_tenant_id: str = Header(default="default")) -> dict[str, Any]:
        body.tenant = x_tenant_id
        resp = await eng(request).complete(body)
        return resp.model_dump()

    @app.get("/v1/traces/{trace_id}")
    async def trace(trace_id: str, request: Request) -> dict[str, Any]:
        spans = eng(request).tracer.get(trace_id)
        if not spans:
            raise HTTPException(404, "trace not found")
        return {"trace_id": trace_id, "spans": spans}

    @app.get("/v1/providers")
    async def providers(request: Request) -> dict[str, Any]:
        g = eng(request).gateway
        return {
            "providers": {n: {"circuit": g.breaker(n).state.value} for n in g._slots},
            "models": {m: {"provider": c.provider, "latency_ewma_ms": round(g.latency.get(m), 2),
                           "quality": c.quality} for m, c in g.models.items()},
        }

    return app


app = create_app()
