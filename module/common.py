import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pyworld as pw
import numpy as np


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class AdaptiveChannelNorm(nn.Module):
    def __init__(self, channels, pitch_emb, eps=1e-6):
        super().__init__()
        self.shift = nn.Conv1d(pitch_emb, channels, 1, 1, 0)
        self.scale = nn.Conv1d(pitch_emb, channels, 1, 1, 0)
        self.eps = eps

    def forward(self, x, p):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale(p) + self.shift(p)
        return x



class ConvNeXt1d(nn.Module):
    def __init__(self, channels=512, hidden_channels=1536, kernel_size=7, scale=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, padding='same', groups=channels)
        self.norm = ChannelNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res


class AdaptiveConvNeXt1d(nn.Module):
    def __init__(self, channels=512, hidden_channels=1536, pitch_emb=128, kernel_size=7, scale=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, padding='same', groups=channels)
        self.norm = AdaptiveChannelNorm(channels, pitch_emb)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x, p):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x, p)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res


class UNetLayer(nn.Module):
    def __init__(self, channels=512, hidden_channels=1536, condition_emb=128, kernel_size=7, scale=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, padding='same', groups=channels)
        self.norm = AdaptiveChannelNorm(channels, condition_emb)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x, p):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x, p)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res



# helper functions
def match_features(source, reference, k=4, alpha=0.0):
    input_data = source
    with torch.no_grad():
        # source: [N, 768, Length], reference: [N, 768, Length]
        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        source_norm = torch.norm(source, dim=2, keepdim=True)
        reference_norm = torch.norm(reference, dim=2, keepdim=True)
        cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
        best = torch.topk(cos_sims, k, dim=2)
    result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
    result = result.transpose(1, 2)
    return result * (1-alpha) + input_data * alpha



def compute_f0(wf, sample_rate=16000, segment_size=256, f0_min=20, f0_max=1100):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = pw.stonemask(signal, _f0, t, sample_rate)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [compute_f0(wave[0], sample_rate, segment_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs
