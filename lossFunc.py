import torch
import torch.nn.functional as F


def nnpu_loss(z, t, prior, weights=None, sur_loss='sigmoid', gamma=1.0, beta=0.0):
    if weights is None:
        weights = torch.ones(t.size(0), device=t.device)
    
    if sur_loss == 'sigmoid':
        loss = (lambda x: torch.sigmoid(-x))
    elif sur_loss == 'logistic':
        loss = (lambda x: F.softplus(-x))
    else:
        raise ValueError('Invalid surrogate loss.')
    
    positive, unlabeled = (t == 1).float(), (t == -1).float()
    
    n_positive, n_unlabeled = max([1., positive.sum()]), max([1., unlabeled.sum()])
    y_positive = loss(z).view(-1) * weights
    y_unlabeled = loss(-z).view(-1) * weights
    positive_risk = torch.sum(prior * positive * y_positive / n_positive)
    negative_risk = torch.sum((unlabeled / n_unlabeled - prior * positive / n_positive) * y_unlabeled)
    
    if negative_risk.data < -beta:
        risk = -gamma * negative_risk
    else:
        risk = positive_risk + negative_risk
    
    return risk