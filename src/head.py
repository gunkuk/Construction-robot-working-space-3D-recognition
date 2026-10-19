"""Trainable light multi-scale readout; the encoder is owned and frozen by the caller."""
import torch
from torch import nn


class MultiScaleSegHead(nn.Module):
    def __init__(self, channels, width=64, hidden=128, classes=18, dropout=0.1):
        super().__init__()
        self.channels = tuple(channels)
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(c, width), nn.LayerNorm(width), nn.GELU())
                                         for c in self.channels)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, features, inverses):
        if len(features) != len(self.projections) or len(inverses) != len(features) - 1:
            raise ValueError('Head input hierarchy does not match checkpoint channels')
        fused = self.projections[0](features[0])
        for index in range(1, len(features)):
            inverse = inverses[index - 1]
            if inverse.numel() != features[index].shape[0]:
                raise ValueError('Wrong pooling inverse length')
            fused = (fused[inverse] + self.projections[index](features[index])) * (2 ** -0.5)
        return self.classifier(fused)


def frozen_hierarchy(point):
    features, inverses = [], []
    while True:
        features.append(point.feat.detach())
        if 'pooling_parent' not in point:
            break
        inverses.append(point.pooling_inverse.detach().long())
        point = point.pooling_parent
    return features, inverses
