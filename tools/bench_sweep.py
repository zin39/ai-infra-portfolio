#!/usr/bin/env python3
"""Concurrency sweep against any OpenAI-compatible /v1/chat/completions endpoint.

Stdlib only, so it runs on a bare GPU box. For each concurrency level it fires N
streaming requests at once and reports per-request TTFT, per-user decode speed,
and aggregate throughput. The *worst* user is reported on purpose: an average
hides the user who is starving, and that user is the one who files the ticket.

Usage:
  python3 bench_sweep.py --url http://localhost:8000 --model my-model \
      --concurrency 1,4,8,16 --max-tokens 256 --prompt-words 2000
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Result:
    ttft_s: float
    total_s: float
    tokens: int

    @property
    def decode_tps(self) -> float:
        """Tokens after the first one, divided by time after the first one."""
        span = self.total_s - self.ttft_s
        return (self.tokens - 1) / span if self.tokens > 1 and span > 0 else 0.0


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: float = 600) -> Result:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft, tokens = None, 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                tokens += 1  # one SSE chunk ~ one token for most servers
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return Result(ttft if ttft is not None else total, total, tokens)


def summarise(results: list[Result], wall_s: float) -> dict:
    tps = [r.decode_tps for r in results]
    return {
        "n": len(results),
        "ttft_p50_s": round(statistics.median(r.ttft_s for r in results), 3),
        "ttft_max_s": round(max(r.ttft_s for r in results), 3),
        "decode_worst_tps": round(min(tps), 1),
        "decode_avg_tps": round(statistics.mean(tps), 1),
        "aggregate_tps": round(sum(r.tokens for r in results) / wall_s, 1) if wall_s > 0 else 0.0,
    }


def _safe_request(url: str, model: str, prompt: str, max_tokens: int) -> Result | None:
    """A failed request is counted, not fatal: overload errors are part of the result."""
    try:
        return one_request(url, model, prompt, max_tokens)
    except Exception as exc:  # noqa: BLE001
        print(f"request failed: {exc}", file=sys.stderr)
        return None


def sweep(url: str, model: str, levels: list[int], max_tokens: int, prompt: str) -> list[dict]:
    rows = []
    for c in levels:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            outcomes = list(pool.map(lambda _: _safe_request(url, model, prompt, max_tokens), range(c)))
        ok = [r for r in outcomes if r is not None]
        stats = summarise(ok, time.perf_counter() - start) if ok else {"n": 0}
        row = {"concurrency": c, **stats, "errors": len(outcomes) - len(ok)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", default="1,4,8,16")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt-words", type=int, default=200, help="rough prompt length to stress prefill")
    a = p.parse_args(argv)
    prompt = " ".join(["benchmark"] * a.prompt_words) + "\nSummarise the text above in one paragraph."
    sweep(a.url, a.model, [int(x) for x in a.concurrency.split(",")], a.max_tokens, prompt)


if __name__ == "__main__":
    main()
