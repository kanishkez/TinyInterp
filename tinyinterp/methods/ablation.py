import torch
from typing import Dict, Callable

def get_ablation_hook(ablation_type: str = "zero", mean_tensor: torch.Tensor = None) -> Callable:
    def hook_fn(module, input, output):
        if ablation_type == "zero": return torch.zeros_like(output)
        elif ablation_type == "mean": return mean_tensor.expand_as(output).to(output.device)
        else: raise ValueError(f"Unknown type {ablation_type}")
    return hook_fn

def run_with_ablation(model, *args, component_names: list, ablation_type: str = "zero", mean_cache: Dict[str, torch.Tensor] = None, **kwargs):
    fwd_hooks = []
    for name in component_names:
        mean_t = mean_cache[name] if mean_cache and name in mean_cache else None
        fwd_hooks.append((name, get_ablation_hook(ablation_type, mean_t)))
    return model.run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
