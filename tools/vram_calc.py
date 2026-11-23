#!/usr/bin/env python3
"""Back-of-envelope GPU sizing for LLM serving.

Answers the three questions I ask before deploying any model:
  1. Do the weights fit?           weights = params * bits / 8
  2. How many tokens of KV fit?    kv/token = 2 * layers * kv_heads * head_dim * kv_bytes
  3. What is the decode ceiling?   single-stream tok/s <= memory bandwidth / bytes read per token

Usage:
  python3 vram_calc.py --params 8 --bits 16 --layers 32 --kv-heads 8 --head-dim 128 \
      --vram-gb 96 --bandwidth-gbs 1792 --util 0.9
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GB = 1e9


@dataclass
class Plan:
    weights_gb: float
    kv_bytes_per_token: float
    kv_budget_gb: float
    max_kv_tokens: int
    decode_ceiling_tps: float


def weights_gb(params_b: float, bits: float) -> float:
    """Weight size in GB for `params_b` billion parameters at `bits` per weight."""
    return params_b * 1e9 * bits / 8 / GB


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, kv_bits: float = 16) -> float:
    """K and V for every layer, for one token (standard multi-head / GQA attention)."""
    return 2 * layers * kv_heads * head_dim * kv_bits / 8


def plan(
    params_b: float,
    bits: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    vram_gb: float,
    bandwidth_gbs: float,
    util: float = 0.9,
    kv_bits: float = 16,
    overhead_gb: float = 2.0,
    active_params_b: float | None = None,
) -> Plan:
    """Size a deployment. `active_params_b` is for MoE models (params touched per token)."""
    if not 0 < util <= 1:
        raise ValueError("util must be in (0, 1]")
    w = weights_gb(params_b, bits)
    kv_tok = kv_bytes_per_token(layers, kv_heads, head_dim, kv_bits)
    budget = vram_gb * util - w - overhead_gb
    max_tokens = max(0, int(budget * GB // kv_tok)) if kv_tok else 0
    read_per_token_gb = weights_gb(active_params_b or params_b, bits)
    ceiling = bandwidth_gbs / read_per_token_gb if read_per_token_gb else 0.0
    return Plan(w, kv_tok, max(budget, 0.0), max_tokens, ceiling)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=float, required=True, help="total parameters, billions")
    p.add_argument("--active-params", type=float, help="MoE: active parameters per token, billions")
    p.add_argument("--bits", type=float, default=16, help="bits per weight (16, 8, 4.5 for Q4_K_M ...)")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--kv-bits", type=float, default=16, help="16 = fp16/bf16 KV, 8 = fp8 KV")
    p.add_argument("--vram-gb", type=float, required=True)
    p.add_argument("--bandwidth-gbs", type=float, required=True, help="GPU memory bandwidth, GB/s")
    p.add_argument("--util", type=float, default=0.9, help="fraction of VRAM the engine may use")
    p.add_argument("--overhead-gb", type=float, default=2.0, help="activations, CUDA graphs, workspace")
    a = p.parse_args(argv)
    r = plan(a.params, a.bits, a.layers, a.kv_heads, a.head_dim, a.vram_gb, a.bandwidth_gbs,
             a.util, a.kv_bits, a.overhead_gb, a.active_params)
    print(f"weights            {r.weights_gb:8.2f} GB")
    print(f"KV per token       {r.kv_bytes_per_token / 1024:8.1f} KiB")
    print(f"KV budget          {r.kv_budget_gb:8.2f} GB")
    print(f"max KV tokens      {r.max_kv_tokens:8,d}")
    print(f"decode ceiling     {r.decode_ceiling_tps:8.1f} tok/s (single stream, bandwidth bound)")
    if r.max_kv_tokens == 0:
        print("!! weights do not fit: quantise, add GPUs, or offload")


if __name__ == "__main__":
    main()
