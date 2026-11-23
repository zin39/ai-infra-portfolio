# Running 320B and 753B-parameter models on hardware that "can't" hold them

**Stack:** llama.cpp (CUDA, RPC backend), GGUF quantisation, 2× H200 servers
**Outcome:** a 320B MoE split across two servers, with **2.3× faster generation** from MTP speculative decoding.
A 753B MoE on a single 2-GPU server, **3.5× faster** than the baseline configuration.

## 320B MoE across two servers

- Check the network first. Measured server-to-server bandwidth was **2.3–3.45 Gbps**. That is far too slow for
  tensor parallelism, which exchanges activations every layer, but fine for **pipeline-style layer splitting** over llama.cpp RPC.
- Speculative decoding with the model's built-in multi-token-prediction (MTP) head:

| Config | Decode |
|---|---|
| Baseline | 8.58 tok/s |
| MTP, draft length 3 | **19.3–23.5 tok/s (≈2.3×)** |
| MTP, draft length 4 | worse: acceptance fell to 24%, so wasted drafts cost more than they saved |

## 753B MoE (455 GB at 4-bit) on one 2× H200 server

The weights don't fit in 282 GB of VRAM. MoE helps because only a few experts are active per token, so the
**expert weights of some layers can live in CPU RAM** while attention and shared weights stay on GPU.

- Offloaded the experts of 34 layers to CPU (`--n-cpu-moe`), and split the rest unevenly across the two GPUs (`-ts 54,24`)
  to balance memory.
- Matched CPU threads to the container's real core limit (24 cores in the cgroup, not the host's count).
- **4.0 → 13.98 tok/s (3.5×).**

## Takeaway

Before buying hardware, check three things. What is the bandwidth between the parts? Which weights are touched per token?
Where is the real CPU limit? Answering those turned "needs a bigger cluster" into "runs on what we have".
