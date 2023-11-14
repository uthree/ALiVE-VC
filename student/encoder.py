import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import ConvNeXt1d

class Encoder(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            segment_size=256,
            internal_channels=512,
            hidden_channels=1536,
            num_layers=4,
            output_dim=768,
            max_f0=4096
            ):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft//2+1, internal_channels, 1, 1, 0)
        self.mid_layers = nn.Sequential(*[
                ConvNeXt1d(internal_channels, hidden_channels, scale=1/num_layers)
            ])
        self.to_f0 = nn.Conv1d(internal_channels, max_f0, 1, 1, 0)
        self.to_con = nn.Conv1d(internal_channels, output_dim, 1, 1, 0)
        self.segment_size = segment_size
        self.n_fft = n_fft

    def forward(self, wave):
        x = torch.stft(wave, self.n_fft, self.segment_size, return_complex=True).abs()[:, :, 1:]
        return self.encode_spectrogram(x)

    def encode_spectrogram(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        f0 = self.to_f0(x)
        con = self.to_con(x)
        return con, f0
