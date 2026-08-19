import torch
from tinyinterp import HookedTransformer, HookedTransformerConfig, activation_patch, run_with_ablation, act_x_grad_attribution, SparseAutoencoder, steer_feature, js_divergence, cka_similarity
def main():
    config = HookedTransformerConfig(d_model=128, n_heads=4, d_mlp=512, n_layers=2, vocab_size=1000, max_seq_len=64)
    model = HookedTransformer(config)
    model.eval()
    
    clean_tokens = torch.tensor([[10, 20, 30, 40, 50], [15, 25, 35, 45, 55]])
    corrupt_tokens = torch.tensor([[11, 21, 31, 41, 51], [16, 26, 36, 46, 56]])
    
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
    print(f"Cached {len(clean_cache)} hook points!")
        
    patch_dict = {"blocks.0.hook_resid_post": corrupt_cache["blocks.0.hook_resid_post"]}
    with torch.no_grad(): patched_logits = activation_patch(model, (clean_tokens,), patch_dict)
    print(f"Patched L1 diff: {(patched_logits - clean_logits).abs().mean().item():.4f}")

    with torch.no_grad(): ablated_logits = run_with_ablation(model, clean_tokens, component_names=["blocks.1.hook_mlp_out"], ablation_type="zero")
    print(f"Ablated L1 diff: {(ablated_logits - clean_logits).abs().mean().item():.4f}")

    attr = act_x_grad_attribution(model, (clean_tokens,), lambda l: l[0, 0, 10], layer_pattern="hook_resid_post")
    for k, v in attr.items(): print(f" - {k} attribution shape: {v.shape}")

    sae = SparseAutoencoder(d_model=128, d_sae=512)
    with torch.no_grad(): steered_logits = steer_feature(model, clean_tokens, hook_name="blocks.0.hook_resid_post", sae=sae, feature_idx=42, scale=10.0, mode="add")
    print(f"Steered L1 diff: {(steered_logits - clean_logits).abs().mean().item():.4f}")

    print(f"JS Divergence: {js_divergence(clean_logits, corrupt_logits).item():.4f}")
    print(f"CKA Similarity: {cka_similarity(clean_cache['blocks.1.hook_resid_post'], corrupt_cache['blocks.1.hook_resid_post']):.4f}")

if __name__ == "__main__": main()
