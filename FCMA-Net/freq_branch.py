import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    
    def __init__(self):
        super(HaarDWT, self).__init__()
        
        ll_filter = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh_filter = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hl_filter = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hh_filter = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        
        self.register_buffer('ll_filter', ll_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', lh_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', hl_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', hh_filter.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        if H % 2 != 0 or W % 2 != 0:
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[2], x.shape[3]
        
        dwt_channels = []
        
        for c in range(C):
            channel_x = x[:, c:c+1, :, :]
            
            ll = F.conv2d(channel_x, self.ll_filter, stride=2, padding=0)
            lh = F.conv2d(channel_x, self.lh_filter, stride=2, padding=0)
            hl = F.conv2d(channel_x, self.hl_filter, stride=2, padding=0)
            hh = F.conv2d(channel_x, self.hh_filter, stride=2, padding=0)
            
            channel_dwt = torch.cat([ll, lh, hl, hh], dim=1)
            dwt_channels.append(channel_dwt)
        
        dwt_output = torch.cat(dwt_channels, dim=1)
        
        return dwt_output


class FreqBranch(nn.Module):
    
    def __init__(self, in_ch=12, embed_dim=256, output_tokens=False):
        super(FreqBranch, self).__init__()
        self.output_tokens = output_tokens
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        if output_tokens:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(16)
            self.projection = nn.Linear(128, embed_dim)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.projection = nn.Linear(128, embed_dim)
        
    def forward(self, dwt_input):
        features = self.conv_layers(dwt_input)
        
        if self.output_tokens:
            pooled = self.adaptive_pool(features)
            B, C, H, W = pooled.shape
            tokens = pooled.view(B, C, H * W).transpose(1, 2)
            freq_tokens = self.projection(tokens)
            return freq_tokens
        else:
            pooled = self.global_pool(features)
            pooled = pooled.view(pooled.size(0), -1)
            freq_embed = self.projection(pooled)
            return freq_embed


def test_freq_branch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dwt = HaarDWT().to(device)
    freq_branch_embed = FreqBranch(in_ch=12, embed_dim=256, output_tokens=False).to(device)
    freq_branch_tokens = FreqBranch(in_ch=12, embed_dim=768, output_tokens=True).to(device)
    
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        dwt_output = dwt(test_input)
        print(f"DWT output shape: {dwt_output.shape}")
        
        freq_embed = freq_branch_embed(dwt_output)
        print(f"Freq embed shape: {freq_embed.shape}")
        
        freq_tokens = freq_branch_tokens(dwt_output)
        print(f"Freq tokens shape: {freq_tokens.shape}")
        
        expected_dwt_shape = (batch_size, 12, 112, 112)
        expected_embed_shape = (batch_size, 256)
        expected_tokens_shape = (batch_size, 256, 768)
        
        assert dwt_output.shape == expected_dwt_shape
        assert freq_embed.shape == expected_embed_shape
        assert freq_tokens.shape == expected_tokens_shape
        
        print("✓ All shape tests passed")
        print("✓ Freq branch test successful")


if __name__ == "__main__":
    test_freq_branch()
