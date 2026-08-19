import torch
from tinyinterp import HookedTransformer, HookedTransformerConfig, HookedGPT2, HookedGPT2Config, logit_lens, direct_logit_attribution, activation_patch, run_with_ablation, act_x_grad_attribution, train_probe, js_divergence, cka_similarity, SparseAutoencoder, steer_feature
def test_all():
    cfg = HookedTransformerConfig(d_model=64, n_heads=2, d_mlp=128, n_layers=2, vocab_size=100, max_seq_len=16)
    model = HookedTransformer(cfg).eval()
    tokens = torch.tensor([[10, 20, 30], [15, 25, 35]])
    logits, cache = model.run_with_cache(tokens)
    assert "blocks.0.hook_resid_post" in cache
    logit_lens(model, cache, "hook_resid_post")
    direct_logit_attribution(model, cache, "hook_mlp_out")
    
    corrupt_tokens = torch.tensor([[11, 21, 31], [16, 26, 36]])
    _, corrupt_cache = model.run_with_cache(corrupt_tokens)
    patched_logits = activation_patch(model, (tokens,), {"blocks.0.hook_resid_post": corrupt_cache["blocks.0.hook_resid_post"]})
    run_with_ablation(model, tokens, component_names=["blocks.1.hook_mlp_out"], ablation_type="zero")
    
    act_x_grad_attribution(model, (tokens,), lambda l: l[0, -1, 10], "hook_resid_post")
    train_probe(cache["blocks.0.hook_resid_post"].reshape(-1, 64).detach(), torch.tensor([0, 1, 0, 1, 0, 1]), epochs=1)
    
    js_divergence(logits, patched_logits)
    cka_similarity(cache["blocks.0.hook_resid_post"], corrupt_cache["blocks.0.hook_resid_post"])
    steer_feature(model, tokens, hook_name="blocks.0.hook_resid_post", sae=SparseAutoencoder(64, 128), feature_idx=5, scale=2.0)
    
    gpt_cfg = HookedGPT2Config(d_model=64, n_heads=2, d_mlp=128, n_layers=2, vocab_size=100, max_seq_len=16)
    HookedGPT2(gpt_cfg).run_with_cache(tokens)
    print("\n✅ ALL SMOKE TESTS PASSED!")
if __name__ == "__main__": test_all()
