# TinyInterp 🔬

A lightweight, from-scratch Python library for mechanistic interpretability. Built entirely with PyTorch, TinyInterp provides the core primitives (hooking, caching, intervening) and standard techniques (Logit Lens, Patching, Probing) without the bloat of heavy abstractions.

## Features

- **TransformerLens-style API:** Seamlessly extract intermediate activations using `logits, cache = model.run_with_cache(tokens)`.
- **Scratch-Built Hooking:** Non-destructively inject callbacks anywhere in the network using `HookPoint` and `HookedModule`.
- **Pre-trained Support:** Load real weights like the canonical 124M GPT-2 using `HookedGPT2.from_pretrained("gpt2")`.
- **Core Techniques:**
  - **Logit Lens & DLA:** Project intermediate residual streams directly to vocabulary logits.
  - **Activation Patching:** Resample/causally trace activations across different prompts.
  - **Ablation:** Zero or mean-ablate specific nodes or components.
  - **Attribution:** Compute Activation $\times$ Gradient attribution for specific layers against a custom target function.
  - **SAE Feature Steering:** Load Sparse Autoencoders and perform feature surgery (add, set, clamp).
  - **Metrics:** Compute JS Divergence and Centered Kernel Alignment (CKA) between representations.
  - **Probing:** Train linear probes on intermediate representations.

## Quickstart

```python
import torch
from tinyinterp import HookedGPT2, logit_lens

# Load a real pre-trained model!
model = HookedGPT2.from_pretrained("gpt2")
model.eval()

# Run with cache
tokens = torch.tensor([[10, 20, 30, 40, 50]])
logits, cache = model.run_with_cache(tokens)

# Inspect the residual stream halfway through the network
lens = logit_lens(model, cache, layer_pattern="blocks.6.hook_resid_post")
print(lens["blocks.6.hook_resid_post"].shape)
```

## Demos & Smoke Tests
Check out `demo.py`, `demo_gpt2.py`, and `smoke_test.py` for comprehensive examples of every technique in action!
