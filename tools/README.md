# Tools

Two small, dependency-free Python tools (Python 3.10+). I wrote them from scratch for this portfolio, based on the
methods I use at work.

## `vram_calc.py`: will it fit, and how fast can it go?

Answers three questions before you deploy a model:

1. **Do the weights fit?** `weights = parameters × bits ÷ 8`
2. **How much context fits?** `KV per token = 2 × layers × KV heads × head dim × KV bytes`, and whatever VRAM is left after the weights goes to the KV cache.
3. **What is the speed ceiling?** For one user, every active weight is read once per generated token, so
   `max tok/s ≈ memory bandwidth ÷ bytes read per token`.

```console
$ python3 tools/vram_calc.py --params 8 --bits 16 --layers 32 --kv-heads 8 --head-dim 128 \
      --vram-gb 96 --bandwidth-gbs 1792
weights               16.00 GB
KV per token          128.0 KiB
KV budget             68.40 GB
max KV tokens       521,850
decode ceiling        112.0 tok/s (single stream, bandwidth bound)
```

On that GPU (RTX PRO 6000 Blackwell) I measured **82.7 tok/s**, which is 74% of the ceiling and a healthy result.
If a measurement comes in far below the ceiling, the setup is misconfigured.

| Option | Meaning |
|---|---|
| `--params` | Total parameters, in billions |
| `--active-params` | MoE models only: parameters used per token, in billions. This sets the speed ceiling |
| `--bits` | Bits per weight: 16 (FP16/BF16), 8, 4, or about 4.5 for llama.cpp Q4_K_M |
| `--layers`, `--kv-heads`, `--head-dim` | From the model's `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim`) |
| `--kv-bits` | 16 for a normal KV cache, 8 for an FP8 KV cache (twice the context) |
| `--vram-gb`, `--bandwidth-gbs` | GPU memory size and bandwidth, from the spec sheet |
| `--util` | Fraction of VRAM the inference engine may use (vLLM's `--gpu-memory-utilization`). Default 0.9 |
| `--overhead-gb` | Reserve for activations, CUDA graphs and workspace. Default 2 |

This is a back-of-envelope estimate. It ignores attention variants such as MLA and sliding windows, engine-specific
overheads, and compute-bound prefill.

## `bench_sweep.py`: how does it behave under load?

Fires *N* streaming requests at once at any **OpenAI-compatible** `/v1/chat/completions` endpoint (vLLM, SGLang,
llama.cpp server, Ollama, LiteLLM) and repeats this for each concurrency level. Each level prints one JSON line:

```console
$ python3 tools/bench_sweep.py --url http://localhost:8000 --model my-model --concurrency 1,8
{"concurrency": 1, "n": 1, "ttft_p50_s": 0.041, "ttft_max_s": 0.041, "decode_worst_tps": 81.9, "decode_avg_tps": 81.9, "aggregate_tps": 80.1, "errors": 0}
{"concurrency": 8, "n": 8, "ttft_p50_s": 0.093, "ttft_max_s": 0.12, "decode_worst_tps": 70.2, "decode_avg_tps": 72.5, "aggregate_tps": 561.4, "errors": 0}
```
*(The numbers above show the output format. They are not from a specific run.)*

| Field | Meaning |
|---|---|
| `ttft_p50_s`, `ttft_max_s` | Time to first token, median and worst. This is what users feel as "lag" |
| `decode_worst_tps` | Generation speed of the **slowest** user. Averages hide the user who is starving |
| `decode_avg_tps` | Mean generation speed per user |
| `aggregate_tps` | Total tokens per second across all users, which is the capacity number |
| `errors` | Requests that failed or timed out. These are counted instead of crashing the run |

| Option | Default | Meaning |
|---|---|---|
| `--url` | (required) | Base URL of the server, without `/v1` |
| `--model` | (required) | Model name the server expects |
| `--concurrency` | `1,4,8,16` | Comma-separated concurrency levels |
| `--max-tokens` | `256` | Tokens to generate per request |
| `--prompt-words` | `200` | Rough prompt length. Raise it to stress prefill |

Tokens are counted as streamed content chunks. Most servers send one token per chunk, so this is a close
approximation.

## Tests

```bash
pip install pytest
python3 -m pytest tools/tests -v
```

The tests check the sizing formulas against known model shapes. They also start a fake streaming server on
localhost, so the load tester is tested end to end without a GPU or network access. CI runs them on every push.
