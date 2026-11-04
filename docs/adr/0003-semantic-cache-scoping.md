# ADR-0003: Scope the semantic cache by context, not just similarity

**Status:** accepted. Prompted by a bug the test suite caught.

## Context
The first version embedded the **whole prompt** and returned a cached answer when cosine similarity was ≥ 0.95. In RAG, the retrieved context is 90%+ of the prompt text. Two *different* questions over the same passages ("How long do card refunds take?" vs. "How often are keys rotated?") embedded almost identically, so the second question got the first question's answer. The eval harness surfaced this as `answer_match = 0.5`.

## Decision
- The **semantic key** is only the question: `metadata["semantic_key"]`, or the final user turn by default.
- Everything else must match **exactly** and goes into the scope hash: tenant, model, all prior messages, and `metadata["semantic_scope"]`. The RAG pipeline sets the scope to the ids of the retrieved chunks.
- Only deterministic requests are cached (`temperature == 0`, no tools).

## Consequences
- A paraphrased question is served from cache only when it retrieved the same passages, which is when the cached answer is actually valid.
- A regression test (`test_semantic_cache_scoped_by_context_not_fooled_by_it`) pins this behaviour.
- The threshold now only has to separate question paraphrases, which is a much easier job than separating whole prompts.
