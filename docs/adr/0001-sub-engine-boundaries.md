# ADR-0001: Sub-engines never import each other

**Status:** accepted

## Context
LLM systems tend to grow into a tangle: retrieval code calls the model client directly, guardrails get bolted onto one endpoint and skipped on another, and cost tracking is spread across features.

## Decision
Gateway, retrieval, guardrails, agents and evals each live in their own package and depend only on `meridian.core`. The main `Engine` is the only module that composes them. The one allowed exception is agents → gateway, because an agent's job is to call a model. Even then it takes the gateway as a constructor argument and doesn't build one itself.

## Consequences
- Each sub-engine has its own tests. None of them needs a running engine.
- Moving a sub-engine into its own service (for example, retrieval on GPU nodes) means writing a client that implements the same interface.
- Composition logic is all in `engine.py`, so changes to request flow happen in one place.
