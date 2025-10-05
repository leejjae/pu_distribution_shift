import torch

def make_labels_from_fixmatch(logits_w, tau=0.95, T=1.0):
    # logits_w: (MuB,)
    p = torch.sigmoid(logits_w / T)
    conf = torch.maximum(p, 1.0 - p)
    mask = (conf >= tau).float()     # FixMatch mask
    target = (p >= 0.5).float()      # 0/1
    return target.detach(), mask