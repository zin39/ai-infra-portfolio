# Serving 1M-token context on two H200s, and the memory leak that wasn't a leak

**Role:** AI Infrastructure Engineer · **Stack:** SGLang (patched), PyTorch, CUDA, 2× NVIDIA H200 NVL (vGPU), Python
**Outcome:** 5.37M tokens served through back-to-back 1M-context requests with **zero out-of-memory crashes**, and single-request 1M prefill **2× faster** than the first working fix.

> Written from my own notes. The employer, customer, hostnames and internal code are left out on purpose.
> Numbers are from my benchmark logs.

## The goal

Serve an open-weight Mixture-of-Experts model with sparse attention at its full **1,000,000-token** context window.
It runs on a 2-GPU H200 server (tensor parallel = 2), with 4-bit weights, an FP8 KV cache and speculative decoding.
It also had to stay fast at normal context lengths.

| Load | Result |
|---|---|
| 1 user | 346 tok/s decode |
| 8 concurrent users | 1,364 tok/s aggregate |
| 32 concurrent users | **2,258 tok/s** aggregate |
| Prefill at ~105K tokens | **16,285 tok/s** |
| Sustained deep context | 485K tokens resident and decoding |

## The symptom

Short and medium prompts were fine. Any request that grew past **590K–700K tokens** got the server killed for exceeding
its GPU memory cap. The KV cache had been sized for 1M tokens, so on paper there was room. Worse, the engine logged
nothing useful before dying.

## Root cause

My first round of memory instrumentation printed nothing at all. It turned out I had patched a code path that this
model never uses, so step one was proving which attention path actually runs. After that, logging the size of every
large temporary allocation gave a clear picture. The culprit was the **sparse-attention indexer**, a small scoring step
that decides which past tokens each new token attends to.

1. The indexer decides between "compute all scores in one shot" and "compute them in chunks" using a memory budget of
   roughly `min(static_cap, 30% of free GPU memory)`. That budget was effectively set while the GPU still had plenty
   of free memory, so it came out at around **12 GB**.
2. The largest single scoring buffer was about **2.2 GiB**, which always fit under that budget. **The chunking path
   never triggered.**
3. But the budget check looked at one buffer at a time, and about **11 sparse-attention layers** each held
   temporaries that grow with context depth. Together they climbed by **+37 GB** as the context deepened, and that
   pushed the server over its cap at 590–700K tokens.
4. The engine's own message about this decision was logged at debug level, which is why it died silently.

So nothing was leaking. Every allocation was individually "within budget", and the problem only showed up at depths
that most tests never reach.

## Fixes, in order

| Version | Change | Result |
|---|---|---|
| v12 | Force small fixed chunks (128 MiB) | No crash, but prefill was 5–25× slower. Profiling found three serialisation points: a print on every call, a free-memory query on every call, and 16–64× micro-slicing that re-read the indexer cache |
| v13 | Larger 2 GiB chunks, the print and the memory query removed | Fast, but the mix of chunk sizes fragmented PyTorch's caching allocator and the server crashed again |
| **v13b** | **1 GiB chunk cap** | **Zero OOMs**, 7,479 tok/s single 1M-token prefill (**2× v12**), peak memory 117.5 of 143 GB |

Before calling it fixed, I ran 8 concurrent 1M-context users (3.88M tokens) and then a 5.37M-token back-to-back
chain. Both passed with no OOM.

## What I'd tell another engineer

- **First prove your instrumentation is on the code path that actually runs.** My first round of logging was silent because it wasn't.
- **"Each allocation fits" is not the same as "the step fits".** Budget the whole step, across all layers.
- **A fix that only works in slow mode is not done.** Profile the fix itself, because v12's slowness came from overhead in the hot path.
- **Test at the context depth you advertise.** A ceiling at 590K is invisible if you only test at 128K.
