import torch
import torch.nn as nn
from typing import Callable

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
    def encode(self, x): return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
    def decode(self, f): return f @ self.W_dec + self.b_dec
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

def get_sae_steering_hook(sae, feature_idx, scale, mode="add"):
    def sae_hook(module, input, output):
        features = sae.encode(output)
        if mode == "add": features[..., feature_idx] += scale
        elif mode in ["set", "clamp"]: features[..., feature_idx] = scale
        modified = sae.decode(features)
        original_features = sae.encode(output)
        original_reconstruction = sae.decode(original_features)
        return output + (modified - original_reconstruction)
    return sae_hook

def steer_feature(model, *args, hook_name, sae, feature_idx, scale, mode="add", **kwargs):
    hook_fn = get_sae_steering_hook(sae, feature_idx, scale, mode)
    return model.run_with_hooks(*args, fwd_hooks=[(hook_name, hook_fn)], **kwargs)
