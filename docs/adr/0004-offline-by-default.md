# ADR-0004: Offline by default, with a deterministic mock model

**Status:** accepted

## Context
If a test suite or demo needs a paid API key, nobody runs it: not CI, not contributors, not reviewers.

## Decision
- `MockProvider` is an *extractive* model. It finds the numbered passages in the prompt, scores sentences by overlap with the question, and answers with citations. It also has `failure_rate` and `latency_ms` settings for chaos testing.
- `HashingEmbedder` uses signed feature hashing over unigrams, bigrams and character trigrams. No model download, and identical output across machines.
- Real providers (Anthropic, OpenAI-compatible) are configuration only. No code changes.

## Consequences
- `git clone && make test && make eval` works anywhere in seconds.
- Retrieval, citation, guardrail and eval behaviour is tested end to end and deterministically.
- The mock doesn't measure real model quality. The eval harness is the tool for that once real providers are configured.
