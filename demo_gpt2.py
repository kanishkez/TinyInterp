import torch
from tinyinterp import HookedGPT2, logit_lens
def main():
    model = HookedGPT2.from_pretrained("gpt2").eval()
    try:
        from transformers import AutoTokenizer
        tokens = AutoTokenizer.from_pretrained("gpt2")("Mechanistic interpretability is", return_tensors="pt")["input_ids"]
    except ImportError:
        tokens = torch.tensor([[10, 20, 30, 40, 50]])
    with torch.no_grad(): logits, cache = model.run_with_cache(tokens)
    lens = logit_lens(model, cache, layer_pattern="blocks.6.hook_resid_post")
    print(f"Successfully cached {len(cache)} points. Top predictions layer 6:")
    top_k = torch.topk(lens["blocks.6.hook_resid_post"][0, -1, :], k=3)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        for idx in top_k.indices: print(f" - {tokenizer.decode([idx.item()])}")
    except: pass
if __name__ == "__main__": main()
