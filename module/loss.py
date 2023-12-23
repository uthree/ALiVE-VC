import torchaudio
import torch
import torch.nn as nn


class LogMelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            sample_rate=48000,
            n_fft=3840,
            hop_length=960,
            eps=1e-4,
            ):
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate,
                n_fft,
                hop_length=hop_length)
        self.eps = eps
    
    def forward(self, x, y):
        x = x.to(torch.float)
        y = y.to(torch.float)

        x = self.to_mel(x)
        y = self.to_mel(y)

        x = torch.log(x + self.eps)
        y = torch.log(y + self.eps)

        x[x.isnan()] = 0
        x[x.isinf()] = 0
        y[y.isnan()] = 0
        y[y.isinf()] = 0

        return (x - y).abs().mean()
