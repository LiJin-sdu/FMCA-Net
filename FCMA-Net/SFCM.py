import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Sfcm(nn.Module):
    
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super(Sfcm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.gamma_patch = nn.Parameter(torch.tensor(1e-3))
        self.gamma_freq = nn.Parameter(torch.tensor(1e-3))
        
        self.patch_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.freq_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.freq_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.freq_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.patch_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.patch_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.patch_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.freq_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.freq_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.patch_q_proj, self.freq_k_proj, self.freq_v_proj,
                      self.freq_q_proj, self.patch_k_proj, self.patch_v_proj,
                      self.patch_out_proj, self.freq_out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, patch_tokens, freq_tokens):
        B, N_patch, C = patch_tokens.shape
        _, N_freq, _ = freq_tokens.shape
        
        patch_q = self.patch_q_proj(patch_tokens)
        freq_k = self.freq_k_proj(freq_tokens)
        freq_v = self.freq_v_proj(freq_tokens)
        
        patch_q = patch_q.view(B, N_patch, self.num_heads, self.head_dim).transpose(1, 2)
        freq_k = freq_k.view(B, N_freq, self.num_heads, self.head_dim).transpose(1, 2)
        freq_v = freq_v.view(B, N_freq, self.num_heads, self.head_dim).transpose(1, 2)
        
        patch_freq_attn = torch.matmul(patch_q, freq_k.transpose(-2, -1))
        patch_freq_attn = patch_freq_attn / math.sqrt(self.head_dim)
        patch_freq_attn = F.softmax(patch_freq_attn, dim=-1)
        patch_freq_attn = self.dropout(patch_freq_attn)
        
        patch_enhanced = torch.matmul(patch_freq_attn, freq_v)
        patch_enhanced = patch_enhanced.transpose(1, 2).contiguous().view(B, N_patch, C)
        patch_enhanced = self.patch_out_proj(patch_enhanced)
        
        freq_q = self.freq_q_proj(freq_tokens)
        patch_k = self.patch_k_proj(patch_tokens)
        patch_v = self.patch_v_proj(patch_tokens)
        
        freq_q = freq_q.view(B, N_freq, self.num_heads, self.head_dim).transpose(1, 2)
        patch_k = patch_k.view(B, N_patch, self.num_heads, self.head_dim).transpose(1, 2)
        patch_v = patch_v.view(B, N_patch, self.num_heads, self.head_dim).transpose(1, 2)
        
        freq_patch_attn = torch.matmul(freq_q, patch_k.transpose(-2, -1))
        freq_patch_attn = freq_patch_attn / math.sqrt(self.head_dim)
        freq_patch_attn = F.softmax(freq_patch_attn, dim=-1)
        freq_patch_attn = self.dropout(freq_patch_attn)
        
        freq_enhanced = torch.matmul(freq_patch_attn, patch_v)
        freq_enhanced = freq_enhanced.transpose(1, 2).contiguous().view(B, N_freq, C)
        freq_enhanced = self.freq_out_proj(freq_enhanced)
        
        gamma_patch_weighted = torch.sigmoid(self.gamma_patch)
        gamma_freq_weighted = torch.sigmoid(self.gamma_freq)
        
        enhanced_patch_tokens = self.patch_norm(patch_tokens + gamma_patch_weighted * patch_enhanced)
        enhanced_freq_tokens = self.freq_norm(freq_tokens + gamma_freq_weighted * freq_enhanced)
        
        return enhanced_patch_tokens, enhanced_freq_tokens


def test_bidirectional_cross_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_attn = Sfcm(hidden_dim=768, num_heads=8).to(device)
    
    batch_size = 4
    num_patch_tokens = 197
    num_freq_tokens = 256
    
    patch_tokens = torch.randn(batch_size, num_patch_tokens, 768).to(device)
    freq_tokens = torch.randn(batch_size, num_freq_tokens, 768).to(device)
    
    print(f"Input shapes:")
    print(f"  patch_tokens: {patch_tokens.shape}")
    print(f"  freq_tokens: {freq_tokens.shape}")
    
    with torch.no_grad():
        enhanced_patch, enhanced_freq = cross_attn(patch_tokens, freq_tokens)
    
    print(f"Output shapes:")
    print(f"  enhanced_patch_tokens: {enhanced_patch.shape}")
    print(f"  enhanced_freq_tokens: {enhanced_freq.shape}")
    
    assert enhanced_patch.shape == patch_tokens.shape
    assert enhanced_freq.shape == freq_tokens.shape
    
    print(f"Gate params:")
    print(f"  gamma_patch: {cross_attn.gamma_patch.item():.6f}")
    print(f"  gamma_freq: {cross_attn.gamma_freq.item():.6f}")
    
    print("✓ Bidirectional cross attention test successful")


if __name__ == "__main__":
    test_bidirectional_cross_attention()



