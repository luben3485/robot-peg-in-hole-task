import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

