# Private AI with no GPU: a 35B model on CPU servers at 33 tok/s

**Stack:** llama.cpp, Intel Xeon (AVX-512), LiteLLM, Mattermost, PostgreSQL + pgvector, LangGraph, Langfuse
**Outcome:** staff got a useful in-house AI assistant without sending internal data to an outside API, on hardware the company already owned.

## Why

Many organisations, like hospitals and clinics, can't paste internal or personal data into a public chatbot.
Self-hosting keeps the data in-house, but GPUs are expensive and were not available for this.

## What I did

- **Ran a 35B-class open-weight MoE model on CPU-only Xeon servers.** The tuning was a build with AVX-512 and threads matched
  to physical cores, speculative decoding, and KV-cache reuse for repeated system prompts. It reached **33 tok/s**, fast enough to chat.
- **One gateway for every model (LiteLLM).** Self-hosted and cloud models sit behind one OpenAI-compatible endpoint with
  fallback, per-user rate limits and cost tracking. Apps never hard-code a provider.
- **Put it where people already work.** A Mattermost bot answers questions, and long support tickets get an AI summary.
- **Assistant platform:** a Python/LangGraph agent with tool use, memory on PostgreSQL + pgvector (vector + full-text search),
  checkpointed tasks that survive crashes, and Langfuse/OpenTelemetry tracing of latency, tokens and cost per request.

## Takeaway

"Private AI" is mostly an infrastructure problem: right-size the model, put a gateway in front, measure everything,
and deliver it in a tool people already open every day.
