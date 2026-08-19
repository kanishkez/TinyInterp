import torch
import torch.nn.functional as F

def js_divergence(p_logits, q_logits):
    p, q = F.softmax(p_logits, dim=-1), F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')

def center_gram(gram):
    n = gram.shape[0]
    H = torch.eye(n, device=gram.device) - torch.ones((n, n), device=gram.device) / n
    return H @ gram @ H

def cka_similarity(x, y):
    x_flat, y_flat = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    gx, gy = center_gram(x_flat @ x_flat.T), center_gram(y_flat @ y_flat.T)
    dot_xy = torch.trace(gx @ gy)
    nx, ny = torch.sqrt(torch.trace(gx @ gx)), torch.sqrt(torch.trace(gy @ gy))
    return (dot_xy / (nx * ny)).item()
