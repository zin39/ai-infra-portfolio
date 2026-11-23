# Benchmarking a GPU fleet, and the tool I built to do it

**Stack:** vLLM, Python, FastAPI, SQLite, Server-Sent Events, SSH · NVIDIA L4, L40S, RTX PRO 6000 Blackwell, H200
**Outcome:** baseline numbers for every GPU type in the fleet, each one checked against its theoretical ceiling,
plus a self-built benchmarking web app with 214 automated tests.

## Why baselines matter

"How fast is this GPU?" has no single answer. It depends on the model, the precision, the prompt length and how many
users share the card. Customers and sales need a number they can rely on, and engineering needs to know whether a
setup is healthy or misconfigured.

My rule is to **compare every measurement with the theoretical ceiling**. Generating a token for one user means reading
every active weight from GPU memory once, so the best possible single-user speed is roughly
*memory bandwidth ÷ bytes of weights read per token*. You can compute that yourself with
[`tools/vram_calc.py`](../tools/vram_calc.py).

| GPU | Model | Measured | Sanity check |
|---|---|---|---|
| RTX PRO 6000 Blackwell | Llama-3.1-8B FP16, 1 user | 82.7 tok/s | **74%** of the 112 tok/s bandwidth ceiling, which is healthy |
| RTX PRO 6000 Blackwell | same model, 64 / 100 concurrent users | 4,427 / 6,423 tok/s total | batching turns spare compute into throughput |
| L4 | 8B, AWQ 4-bit, 1 user | 41 tok/s | quantisation is what makes small cards usable |
| H200 | gpt-oss-120b, 256 concurrent users | 937.5 output tok/s | large MoE at high concurrency |

If a result comes in far below the ceiling, something is wrong, such as the wrong attention backend, a missing kernel,
too little KV-cache memory, or a CPU bottleneck. Finding that out before a customer does is the point of the exercise.

## The benchmarking tool

The old workflow was to SSH into a GPU box, run scripts by hand, and paste numbers into a document. That was slow,
hard to repeat and easy to get wrong. I replaced it with a small internal web app:

- **Job queue (FastAPI + SQLite):** pick a GPU server, a model and a test plan. The app connects over SSH, starts the
  model server, runs a concurrency sweep and stores every result.
- **Live progress:** logs and metrics stream to the browser over Server-Sent Events while the job runs.
- **Reports:** every run produces a PDF report, so results can be shared without screenshots.
- **214 automated tests (pytest):** a change to the harness cannot silently change the numbers it reports.

The core measurement loop is the same idea as [`tools/bench_sweep.py`](../tools/bench_sweep.py) in this repo. It
reports time-to-first-token, per-user speed and total throughput at each concurrency level, and it always reports
the **slowest** user as well as the average.

## Takeaway

A benchmark is only useful if it is repeatable and you know what "good" looks like. Automate the run, store the
raw data, and always put the measured number next to the theoretical one.
