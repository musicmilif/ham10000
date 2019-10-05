import torch
from torch import nn
import torch.nn.functional as F
from .layers import Flatten

import pretrainedmodels


class HAMNet(nn.Module):
    def __init__(self,
                 n_classes,
                 model_name='resnet50'):
        super(HAMNet, self).__init__()
        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        self.mean = self.backbone.mean
        self.std = self.backbone.std
        
        # HACK: work around for this issue https://github.com/Cadene/pretrained-models.pytorch/issues/120
        final_in_features = self.backbone.last_linear.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.final = nn.Linear(final_in_features, n_classes)

    def forward(self, x):
        feature = self.extract_feat(x)
        logits = self.final(feature)
        logits = F.log_softmax(logits, dim=1)
        
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        return x