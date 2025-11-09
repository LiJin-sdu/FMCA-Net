import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Ftma(nn.Module):
    
    def __init__(self, channels, reduction_ratio=16):
        super(Ftma, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        self.norm = nn.LayerNorm(channels)
        
        self.scale1_dim = max(channels // 4, 1)
        self.scale2_dim = max(channels // 2, 1)
        self.scale3_dim = channels
        
        self.scale1_conv = nn.Conv1d(channels, self.scale1_dim, kernel_size=1)
        self.scale1_mlp = nn.Sequential(
            nn.Linear(self.scale1_dim, self.scale1_dim),
            nn.GELU(),
            nn.Linear(self.scale1_dim, self.scale1_dim)
        )
        
        self.scale2_conv = nn.Conv1d(channels, self.scale2_dim, kernel_size=1)
        self.scale2_mlp = nn.Sequential(
            nn.Linear(self.scale2_dim, self.scale2_dim),
            nn.GELU(),
            nn.Linear(self.scale2_dim, self.scale2_dim)
        )
        
        self.scale3_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
        total_dim = self.scale1_dim + self.scale2_dim + self.scale3_dim
        self.projection = nn.Linear(total_dim, channels)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        B, N, C = x.shape
        
        x_norm = self.norm(x)
        
        x_conv1 = x_norm.transpose(1, 2)
        scale1 = self.scale1_conv(x_conv1)
        scale1 = scale1.transpose(1, 2)
        scale1 = self.scale1_mlp(scale1)
        
        x_conv2 = x_norm.transpose(1, 2)
        scale2 = self.scale2_conv(x_conv2)
        scale2 = scale2.transpose(1, 2)
        scale2 = self.scale2_mlp(scale2)
        
        scale3 = self.scale3_mlp(x_norm)
        
        multi_scale = torch.cat([scale1, scale2, scale3], dim=-1)
        projected = self.projection(multi_scale)
        
        channel_weights = projected.mean(dim=1)
        channel_weights = torch.sigmoid(channel_weights)
        channel_weights = channel_weights.unsqueeze(1)
        
        attended_x = x * channel_weights
        output = x + self.alpha * (attended_x - x)
        
        return output
