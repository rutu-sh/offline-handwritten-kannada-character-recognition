import torch
import torch.nn
import torch.nn.functional as F

import numpy as np

def power_iterator(weights, alpha=1, norm_prop=False): 
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device) 
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    prop = identity - alpha * S
    prop = torch.inverse(prop[None, ...])[0] 
    if norm_prop:
        prop = F.normalize(prop, p=1, dim=-1) 
    return prop

def sim_matrix(x, rbf_scale): 
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c))**2).sum(-1) / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device) 
    weights = weights * (~mask).float()
    return weights

def manifold_smoothing(x, alpha, rbf_scale, norm_prop, prop=None): 
    if prop is None:
        weights = sim_matrix(x, rbf_scale) 
        prop = power_iterator(weights, alpha=alpha, norm_prop=norm_prop) 
    return torch.mm(prop, x)

def label_prop(x, labels, nclasses, alpha, rbf_scale, norm_prop, apply_log, prop=None, epsilon=1e-6):
    labels = F.one_hot(labels, nclasses + 1)
    labels = labels[:, :nclasses].float() # the max label is unlabeled if prop is None:
    weights = sim_matrix(x, rbf_scale) 
    prop = power_iterator(weights, alpha=alpha, norm_prop=norm_prop) 
    y_pred = torch.mm(prop, labels)
    if apply_log:
        y_pred = torch.log(y_pred + epsilon)
    return y_pred

