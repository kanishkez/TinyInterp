import torch
from typing import Dict

def logit_lens(model, cache, layer_pattern: str = "hook_resid_post") -> Dict[str, torch.Tensor]:
    logits_dict = {}
    layer_acts = cache.extract_layers(layer_pattern)
    for name, hidden_state in layer_acts.items():
        normalized = model.unembed_norm(hidden_state)
        logits_dict[name] = model.unembed(normalized)
    return logits_dict

def direct_logit_attribution(model, cache, layer_pattern: str = "hook_mlp_out") -> Dict[str, torch.Tensor]:
    dla_dict = {}
    layer_acts = cache.extract_layers(layer_pattern)
    for name, component_out in layer_acts.items():
        dla_dict[name] = model.unembed(component_out)
    return dla_dict
