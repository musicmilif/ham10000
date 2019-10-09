import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _one_hot(index, classes):
    device = index.device
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.0

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, requires_grad=index.requires_grad)

    return mask.to(device).scatter_(1, index, ones.to(device))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        if isinstance(self.weight, torch.Tensor):
            self.weight = torch.ones(input.size(-1))
        assert self.weight.size(0) == input.size(-1)
        device = input.device
        self.weight = self.weight.to(device)

        y = _one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)

        loss = -1 * y * torch.log(logit)
        loss = self.weight * loss * (1 - logit) ** self.gamma

        return loss.sum()
