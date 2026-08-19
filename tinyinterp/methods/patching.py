import torch
from typing import Dict, Callable

def get_patch_hook(cached_tensor: torch.Tensor, seq_idx: int = None) -> Callable:
    def patch_hook(module, input, output):
        if seq_idx is None:
            return cached_tensor.to(output.device)
        else:
            new_output = output.clone()
            new_output[:, seq_idx, :] = cached_tensor[:, seq_idx, :].to(output.device)
            return new_output
    return patch_hook

def activation_patch(model, clean_args: tuple, patch_dict: Dict[str, torch.Tensor], seq_idx: int = None):
    fwd_hooks = []
    for hook_name, cached_tensor in patch_dict.items():
        fwd_hooks.append((hook_name, get_patch_hook(cached_tensor, seq_idx=seq_idx)))
    return model.run_with_hooks(*clean_args, fwd_hooks=fwd_hooks)
