import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceLibrary(nn.Module):
    def __init__(self, num_tokens=128, hubert_dim=768):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(1, hubert_dim, num_tokens))
        self.values = nn.Parameter(torch.randn(1, hubert_dim, num_tokens))
        self.hubert_dim = hubert_dim

    def match(self, query, alpha=0.0):
        N = query.shape[0]
        len_q = query.shape[2]
        len_kv = self.keys.shape[2]
        q = query.transpose(1, 2).expand(N, len_q, self.hubert_dim)
        k = self.keys.transpose(1, 2).expand(N, len_kv, self.hubert_dim)
        v = self.values.transpose(1, 2).expand(N, len_kv, self.hubert_dim)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2) * (1 - alpha) + query * alpha
        return o
