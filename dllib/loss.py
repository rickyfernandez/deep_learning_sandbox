import torch.nn as nn
import torch.nn.functional as F

def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2

def reduce_loss(loss, reduction="mean"):
    return loss.mean() if reduction=="mean" else loss.sum() if reduction=="sum" else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.ε)
