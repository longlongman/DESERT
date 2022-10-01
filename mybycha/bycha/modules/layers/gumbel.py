import torch

def gumbel_softmax_topk(logits, k=1, tau=1, hard=True, dim=-1):
    while True:
        gumbels = -torch.empty_like(logits).exponential_().log()
        #print(gumbels, gumbels.device)
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)
        if (torch.isinf(gumbels).any()) or (torch.isinf(y_soft).any()) or (torch.isnan(y_soft).any()):
            continue
        else:
            break

    index = None
    if hard:
        _, index = torch.topk(y_soft, k)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, index
