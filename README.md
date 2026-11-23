# Karan Raj Gupta: AI Infrastructure Portfolio

[![tests](https://github.com/zin39/ai-infra-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/zin39/ai-infra-portfolio/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

I'm an **AI Infrastructure Engineer** in Kathmandu, Nepal. I run large language models on NVIDIA GPUs
(H200, RTX PRO 6000 Blackwell, L40S, L4) and debug them down to the scheduler and the memory allocator.
I also build the web apps, backends and operations tooling around them.

This repo holds **write-ups of real work**. My employer's code and customer data are private, so the case studies
describe the problem, the method and the measured result, but not the internals. The [`tools/`](tools/) folder has
small utilities I wrote from scratch for this portfolio. They use the same methods, and anyone can run them.

**Website:** https://zin39.github.io/ai-infra-portfolio/

## Highlights

| Work | Result |
|---|---|
| Serving a 1M-token context model on 2× H200 | **2,258 tok/s** at 32 concurrent users · **16,285 tok/s** prefill · **5.37M tokens with zero out-of-memory crashes** |
| Root-causing a crash at 590K+ tokens | Traced a hidden **+37 GB** memory climb in sparse attention. The fix is **2× faster** than the first workaround |
| Fair multi-user scheduling | Slowest user's speed went from **10.8 to 42.4 tok/s (3.9×)** with no loss for a single user |
| Models bigger than the GPUs | 320B model across two servers (**2.3×** with speculative decoding) · 753B model on one server (**3.5×** with CPU expert offload) |
| GPU fleet benchmarking | Baselines for L4, L40S, RTX PRO 6000 and H200, plus a benchmarking web app with **214 automated tests** |
| Private AI with no GPU | 35B model at **33 tok/s** on CPU-only Xeon servers behind a LiteLLM gateway |

## Case studies

| # | Case study | Topics |
|---|---|---|
| 1 | [Serving 1M-token context, and the memory leak that wasn't a leak](case-studies/01-million-token-serving-oom.md) | SGLang, CUDA memory, root-cause analysis |
| 2 | [Stopping long prompts from starving everyone else](case-studies/02-fair-scheduling.md) | Scheduling, prefill vs decode, tail latency |
| 3 | [Running 320B and 753B models on hardware that "can't" hold them](case-studies/03-huge-moe-models.md) | llama.cpp, MoE offload, speculative decoding |
| 4 | [Benchmarking a GPU fleet, and the tool I built to do it](case-studies/04-gpu-benchmarking.md) | vLLM, roofline checks, FastAPI |
| 5 | [Private AI with no GPU](case-studies/05-private-ai-on-cpu.md) | CPU inference, LLM gateway, RAG, observability |
| 6 | [Operations, infrastructure and web apps](case-studies/06-operations-and-web.md) | Monitoring, PostgreSQL HA, Terraform, full-stack apps |

## Tools you can run

You need Python 3.10 or newer. There are no dependencies except `pytest` for the tests.

```bash
git clone https://github.com/zin39/ai-infra-portfolio.git && cd ai-infra-portfolio

# Will a model fit on a GPU, how much context can it hold, and how fast can it possibly go?
python3 tools/vram_calc.py --params 8 --bits 16 --layers 32 --kv-heads 8 --head-dim 128 \
    --vram-gb 96 --bandwidth-gbs 1792

# Load-test any OpenAI-compatible server (vLLM, SGLang, llama.cpp, Ollama ...)
python3 tools/bench_sweep.py --url http://localhost:8000 --model my-model --concurrency 1,4,8,16

# Run the tests. The load tester runs against a fake streaming server, so no GPU is needed.
pip install pytest && python3 -m pytest tools/tests
```

See [`tools/README.md`](tools/README.md) for every option, example output and how to read the results.

## Other public code

- [terraform-vcenter-vm-portal](https://github.com/zin39/terraform-vcenter-vm-portal): self-service VM provisioning on vCenter (Terraform, web portal, approval workflow)
- [wordpress-aws](https://github.com/zin39/wordpress-aws): AWS VPC and EC2 infrastructure as code, deployed by GitHub Actions
- [openstack_dashboard](https://github.com/zin39/openstack_dashboard): React + Express web UI for OpenStack

## A note on confidentiality

All numbers come from my own benchmark logs. Employer, customer and vendor names, hostnames, internal code and anything
security-sensitive are left out on purpose. I'm happy to go into more depth in an interview, within what my agreements allow.

## Contact

**Email:** guptakaran12000@gmail.com · **LinkedIn:** [karan244-gupta](https://www.linkedin.com/in/karan244-gupta) · Kathmandu, Nepal

Licensed under the [MIT License](LICENSE).
