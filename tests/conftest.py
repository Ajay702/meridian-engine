import pytest

from meridian.core.config import EngineConfig, GatewayConfig, ModelConfig
from meridian.engine import Engine
from meridian.gateway.gateway import ModelGateway
from meridian.gateway.providers.mock import MockProvider

CORPUS = {
    "refunds": "# Refunds\n\n## Processing\nCard refunds take 5 to 10 business days. "
               "Bank transfers take up to 15 business days.\n\n## Eligibility\n"
               "Monthly plans get a full refund within 14 days.",
    "security": "# Security\n\n## Encryption\nData is encrypted at rest with AES-256. "
                "Keys are rotated every 90 days.",
    "sla": "# SLA\n\n## Uptime\nBusiness plans are guaranteed 99.95 percent uptime.",
}


@pytest.fixture
async def engine():
    e = Engine(EngineConfig())
    for doc_id, text in CORPUS.items():
        await e.ingest(doc_id, text, {"team": "support" if doc_id != "security" else "secops"})
    yield e
    await e.aclose()


def two_model_gateway(primary: MockProvider, backup: MockProvider, **cfg) -> ModelGateway:
    models = [
        ModelConfig(name="big", provider=primary.name, input_per_mtok=3, output_per_mtok=15,
                    quality=0.9),
        ModelConfig(name="small", provider=backup.name, input_per_mtok=0.25,
                    output_per_mtok=1.25, quality=0.6),
    ]
    g = ModelGateway(models, GatewayConfig(**cfg))
    g.register_provider(primary)
    g.register_provider(backup)
    return g
