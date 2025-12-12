import torch
import torch.nn.functional as F

def frobenius_loss(model):
    W = model.decoder.weight
    G = W.T @ W
    I = torch.eye(G.size(0), device=G.device)
    
    frob = torch.norm(G - I, p='fro')
    return frob

def srip_loss(model, num_iterations=2):
    W = model.decoder.weight

    WTW = W.T @ W
    I = torch.eye(WTW.shape[0], device=WTW.device, dtype=WTW.dtype)
    M = WTW - I

    height = M.size(0)
    u = F.normalize(M.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)

    for _ in range(num_iterations):
        v = F.normalize(M.T @ u, dim=0, eps=1e-12)
        u = F.normalize(M @ v, dim=0, eps=1e-12)

    sigma = torch.dot(u, M @ v)

    return sigma ** 2

def ortsae_loss(model, delta=1e-8):
    W = model.decoder.weight
    W_norm = W / (W.norm(dim=0, keepdim=True) + delta)
    cos_sim = W_norm.T @ W_norm  

    diag_idx = torch.arange(cos_sim.size(0), device=W.device)
    cos_sim[diag_idx, diag_idx] = 0.0
    max_sim, _ = cos_sim.max(dim=1)

    loss = (max_sim ** 2).mean()
    return loss
