import torch
import torch.nn as nn

class VoiceLibrary(nn.Module):
    def __init__(self, num_tokens=512, hubert_dim=768):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, hubert_dim, num_tokens))
