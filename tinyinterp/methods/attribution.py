import torch
from typing import Callable

def act_x_grad_attribution(model, input_args: tuple, target_fn: Callable, layer_pattern: str = "hook_resid_post"):
    activations = {}
    gradients = {}
    
    def fwd_hook(name):
        def hook(module, input, output):
            if not output.requires_grad: output.requires_grad_(True)
            output.retain_grad()
            activations[name] = output
            output.register_hook(lambda grad: gradients.update({name: grad}))
            return output
        return hook

    fwd_hooks = [(n, fwd_hook(n)) for n, _ in model.named_modules() if layer_pattern in n]
    out = model.run_with_hooks(*input_args, fwd_hooks=fwd_hooks)
    model.zero_grad()
    target_fn(out).backward()
    
    return {n: (activations[n].detach() * gradients[n].detach()) for n in activations if n in gradients}
