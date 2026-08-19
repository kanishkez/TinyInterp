import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..core.hooking import HookPoint, HookedModule

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads
        
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)
        
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_pattern = HookPoint()

    def forward(self, x, freqs_cis, mask):
        B, S, D = x.shape
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_heads, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        xq = self.hook_q(xq)
        xk = self.hook_k(xk)
        xv = self.hook_v(xv)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
            
        attn_pattern = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_pattern = self.hook_pattern(attn_pattern)
        
        out = torch.matmul(attn_pattern, xv)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        
        out = self.hook_z(out)
        return self.wo(out)

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.w2 = nn.Linear(config.d_mlp, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        pre = F.silu(self.w1(x)) * self.w3(x)
        pre = self.hook_post(pre)
        return self.w2(pre)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.hook_attn_out = HookPoint()
        
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU_MLP(config)
        self.hook_mlp_out = HookPoint()
        
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x, freqs_cis, mask):
        normalized_resid = self.attn_norm(x)
        attn_out = self.attn(normalized_resid, freqs_cis, mask)
        attn_out = self.hook_attn_out(attn_out)
        
        x = x + attn_out
        x = self.hook_resid_mid(x)
        
        normalized_resid_mid = self.mlp_norm(x)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = self.hook_mlp_out(mlp_out)
        
        x = x + mlp_out
        x = self.hook_resid_post(x)
        return x

class HookedTransformerConfig:
    def __init__(self, d_model=256, n_heads=8, d_mlp=1024, n_layers=4, vocab_size=50000, max_seq_len=1024):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

class HookedTransformer(HookedModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.hook_embed = HookPoint()
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.unembed_norm = RMSNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.d_model // config.n_heads, config.max_seq_len)
        self.setup_hooks()

    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embed(input_ids)
        x = self.hook_embed(x)
        freqs_cis = self.freqs_cis[:S].to(x.device)
        
        mask = torch.full((S, S), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        
        for block in self.blocks:
            x = block(x, freqs_cis, mask)
            
        x = self.unembed_norm(x)
        logits = self.unembed(x)
        return logits
