import torch
import torch.nn.functional as F
from torch import nn

class cross_entropy(nn.Module):
    def __init__(self, weight=None, reduction='mean',ignore_index=256):
        super(cross_entropy, self).__init__()
        self.weight = weight
        self.ignore_index =ignore_index
        self.reduction = reduction


    def forward(self,input, target):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
