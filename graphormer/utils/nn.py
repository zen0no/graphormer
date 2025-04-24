import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cross_entropy(pred: torch.FloatTensor, target: torch.LongTensor, mask: torch.BoolTensor):
    assert pred.shape == mask.shape

    pred[mask] = -torch.inf

    logits = F.log_softmax(pred, dim=1)

    loss = -logits[torch.arange(logits.shape[0]), target]

    return loss.mean()
