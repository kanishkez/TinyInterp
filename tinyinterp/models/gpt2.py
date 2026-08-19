import torch
import torch.nn as nn
import math
from ..core.hooking import HookPoint, HookedModule

class HookedGPT2Config:
    def __init__(self, d_model=768, n_heads=12, d_mlp=3072, n_layers=12, vocab_size=50257, max_seq_len=1024):
        self.d_model, self.n_heads, self.d_mlp, self.n_layers = d_model, n_heads, d_mlp, n_layers
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.head_dim = d_model // n_heads

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_pattern = HookPoint()

    def forward(self, x, mask=None):
        B, S, D = x.shape
        q, k, v = self.c_attn(x).split(self.config.d_model, dim=2)
        q = self.hook_q(q.view(B, S, self.config.n_heads, self.config.head_dim))
        k = self.hook_k(k.view(B, S, self.config.n_heads, self.config.head_dim))
        v = self.hook_v(v.view(B, S, self.config.n_heads, self.config.head_dim))
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.config.head_dim)
        scores = scores + torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        
        attn_pattern = self.hook_pattern(torch.softmax(scores, dim=-1))
        out = self.hook_z(torch.matmul(attn_pattern, v).transpose(1, 2).contiguous().view(B, S, -1))
        return self.c_proj(out)

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_mlp)
        self.c_proj = nn.Linear(config.d_mlp, config.d_model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        pre = self.hook_pre(self.c_fc(x))
        return self.c_proj(self.hook_post(torch.nn.functional.gelu(pre, approximate="tanh")))

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = GPT2Attention(config)
        self.hook_attn_out = HookPoint()
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = GPT2MLP(config)
        self.hook_mlp_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.ln_1(x))))
        return self.hook_resid_post(x + self.hook_mlp_out(self.mlp(self.ln_2(x))))

class HookedGPT2(HookedModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else HookedGPT2Config()
        self.wte = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.wpe = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.hook_embed, self.hook_pos_embed, self.hook_resid_pre = HookPoint(), HookPoint(), HookPoint()
        self.blocks = nn.ModuleList([GPT2Block(self.config) for _ in range(self.config.n_layers)])
        self.ln_f = nn.LayerNorm(self.config.d_model)
        self.unembed_norm = self.ln_f
        self.unembed = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.unembed.weight = self.wte.weight
        self.setup_hooks()

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.hook_resid_pre(self.hook_embed(self.wte(input_ids)) + self.hook_pos_embed(self.wpe(pos)))
        for block in self.blocks: x = block(x)
        return self.unembed(self.ln_f(x))
        
    @classmethod
    def from_pretrained(cls, model_name="gpt2"):
        import transformers
        print(f"Downloading {model_name} from HuggingFace...")
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config = HookedGPT2Config(
            d_model=hf_model.config.n_embd, n_heads=hf_model.config.n_head,
            d_mlp=hf_model.config.n_embd * 4, n_layers=hf_model.config.n_layer,
            vocab_size=hf_model.config.vocab_size, max_seq_len=hf_model.config.n_positions
        )
        model = cls(config)
        state_dict, hf = model.state_dict(), hf_model.state_dict()
        for i in range(config.n_layers):
            for t in ["weight", "bias"]:
                state_dict[f"blocks.{i}.attn.c_attn.{t}"] = hf[f"transformer.h.{i}.attn.c_attn.{t}"].T if t=="weight" else hf[f"transformer.h.{i}.attn.c_attn.{t}"]
                state_dict[f"blocks.{i}.attn.c_proj.{t}"] = hf[f"transformer.h.{i}.attn.c_proj.{t}"].T if t=="weight" else hf[f"transformer.h.{i}.attn.c_proj.{t}"]
                state_dict[f"blocks.{i}.mlp.c_fc.{t}"] = hf[f"transformer.h.{i}.mlp.c_fc.{t}"].T if t=="weight" else hf[f"transformer.h.{i}.mlp.c_fc.{t}"]
                state_dict[f"blocks.{i}.mlp.c_proj.{t}"] = hf[f"transformer.h.{i}.mlp.c_proj.{t}"].T if t=="weight" else hf[f"transformer.h.{i}.mlp.c_proj.{t}"]
                state_dict[f"blocks.{i}.ln_1.{t}"] = hf[f"transformer.h.{i}.ln_1.{t}"]
                state_dict[f"blocks.{i}.ln_2.{t}"] = hf[f"transformer.h.{i}.ln_2.{t}"]
        for t in ["weight", "bias"]:
            if t == "weight":
                state_dict["wte.weight"], state_dict["wpe.weight"] = hf["transformer.wte.weight"], hf["transformer.wpe.weight"]
                state_dict["ln_f.weight"] = hf["transformer.ln_f.weight"]
            else: state_dict["ln_f.bias"] = hf["transformer.ln_f.bias"]
        model.load_state_dict(state_dict)
        return model
