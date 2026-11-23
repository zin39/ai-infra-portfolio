# Stopping long prompts from starving everyone else (3.9× better worst-case)

**Stack:** SGLang scheduler (patched), Python, 2× H200
**Outcome:** under 8 concurrent users sending 32K-token prompts, the **slowest user's decode speed went from 10.8 to 42.4 tok/s (3.9×)**.
Single-user long-prompt prefill speed stayed the same (11,043 tok/s at 128K).

## The problem

LLM servers do two kinds of work:

- **Prefill** reads the whole prompt. It is heavy and arrives in bursts.
- **Decode** generates one token at a time for everyone already being answered.

When a new 32K-token prompt arrived, the scheduler spent large prefill steps on it while every user mid-answer waited.
Averages looked fine. The **worst** user saw text stall to about 10 tok/s, and that is the user who complains.

## What I changed

1. **Measure the right thing.** My benchmark reports the *worst* user's decode speed, not the mean (see [`tools/bench_sweep.py`](../tools/bench_sweep.py)).
2. **Interleave.** Chunked prefill (32K) plus a prefill-decode interval, so decode steps run between prefill chunks.
3. **Adaptive cap (my patch, behind an env flag).** While anyone is decoding, the prefill chunk is capped to about a
   latency budget × measured prefill speed (≈4K tokens, aligned to 256). When nobody is decoding, prefill runs at full size,
   so a single user loses nothing.

| Metric (8 users × 32K prompts) | Before | After |
|---|---|---|
| Worst-user decode | 10.8 tok/s | **42.4 tok/s** |
| Solo 128K prefill | 11,043 tok/s | 11,043 tok/s (unchanged) |

## Takeaway

Throughput and fairness are separate numbers. Tune for the worst user only while there is contention, and leave solo performance alone.
