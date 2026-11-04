"""Engine configuration: TOML file + environment overrides.

    MERIDIAN_CONFIG=./meridian.toml  meridian serve

Environment variables of the form ``MERIDIAN__SECTION__KEY`` override file values,
e.g. ``MERIDIAN__GATEWAY__CACHE_THRESHOLD=0.9``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str
    kind: str = "mock"  # mock | anthropic | openai_compatible
    models: list[str] = Field(default_factory=list)
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0
    rpm: int = 600
    # mock-only knobs, useful for chaos testing the router
    failure_rate: float = 0.0
    latency_ms: float = 0.0


class ModelConfig(BaseModel):
    name: str
    provider: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    quality: float = 0.5
    """Relative quality score in [0, 1] used by the QUALITY routing policy."""
    context_window: int = 200_000


class GatewayConfig(BaseModel):
    cache_enabled: bool = True
    cache_threshold: float = 0.95
    cache_ttl_s: float = 3600
    cache_capacity: int = 10_000
    max_retries: int = 2
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0
    tenant_budget_usd: float | None = None


class RetrievalConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_dim: int = 512
    top_k: int = 5
    candidate_k: int = 30
    mmr_lambda: float = 0.7
    rrf_k: int = 60


class GuardrailConfig(BaseModel):
    redact_pii: bool = True
    block_injection: bool = True
    injection_threshold: float = 0.6
    max_input_chars: int = 32_000


class EngineConfig(BaseModel):
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [
            ProviderConfig(name="local", kind="mock", models=["mock-large", "mock-small"])
        ]
    )
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(name="mock-large", provider="local", input_per_mtok=3.0,
                        output_per_mtok=15.0, quality=0.9),
            ModelConfig(name="mock-small", provider="local", input_per_mtok=0.25,
                        output_per_mtok=1.25, quality=0.6),
        ]
    )
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> EngineConfig:
        path = path or os.environ.get("MERIDIAN_CONFIG")
        data: dict[str, Any] = {}
        if path:
            data = tomllib.loads(Path(path).read_text())
        _apply_env_overrides(data, os.environ)
        return cls.model_validate(data)


def _apply_env_overrides(data: dict[str, Any], env: os._Environ[str] | dict[str, str]) -> None:
    for key, value in env.items():
        if not key.startswith("MERIDIAN__"):
            continue
        parts = [p.lower() for p in key.removeprefix("MERIDIAN__").split("__") if p]
        if not parts:
            continue
        cursor = data
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
