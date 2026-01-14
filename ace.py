
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveCrossentropy(nn.Module):
    def __init__(self, alpha=0.0, gamma=0.0, label_smoothing=0.0, reduce='none', name='AdaptiveCrossentropy'):
        super(AdaptiveCrossentropy, self).__init__()
        self.eps = 1e-7
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduce = reduce
        assert 0.0 <= self.alpha <= 1.0
        assert self.gamma == 0.0 or self.gamma >= 1.0
        assert 0.0 <= self.label_smoothing <= 0.5
        assert self.reduce in ['none', 'mean', 'sum', 'sum_over_batch_size']

    def forward(self, y_true, y_pred):
        # y_true: (B, C) - One-hot or smoothed
        # y_pred: (B, C) - Sigmoid probabilities (already applied in model)
        
        # Ensure compatible types
        y_true = y_true.type_as(y_pred)
        
        # Clip
        y_true_clip = torch.clamp(y_true, self.label_smoothing, 1.0 - self.label_smoothing)
        y_pred_clip = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        
        # Binary Cross Entropy formula
        loss = -((y_true_clip * torch.log(y_pred_clip + self.eps)) + ((1.0 - y_true_clip) * torch.log(1.0 - y_pred_clip + self.eps)))
        
        if self.alpha > 0.0:
            alpha = torch.ones_like(y_true) * self.alpha
            alpha = torch.where(y_true != 1.0, alpha, 1.0 - alpha)
            loss *= alpha
            
        if self.gamma >= 1.0:
            # Focal loss style weighting
            adaptive_weight = torch.pow(torch.abs(y_true_clip - y_pred_clip), self.gamma)
            loss *= adaptive_weight
            
        if self.reduce == 'mean':
            loss = torch.mean(loss)
        elif self.reduce == 'sum':
            loss = torch.sum(torch.mean(loss, dim=0))
        elif self.reduce == 'sum_over_batch_size':
            loss = torch.sum(loss)
            
        return loss
