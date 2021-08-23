import torch
import torch.nn as nn
import torch.nn.functional as F


class CELossWithLS(nn.Module):
    '''
    label smoothing cross-entropy loss for captioning
    '''
    def __init__(self, classes=None, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).permute(0,1,2).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.smoothing / self.cls

        logs = self.log_softmax(logits[target!=self.ignore_index])
        pt = torch.exp(logs)
        return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()

