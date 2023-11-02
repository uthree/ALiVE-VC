import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size -1)*dilation)/2)


class F0Encoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)
        self.c2 = nn.Conv1d(output_dim, output_dim, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x


class AmplitudeEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, output_dim, 1, 1, 0)

    def forward(self, amp):
        return self.c1(amp)



class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])

        for d in dilation:
            self.convs1.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d))))
            self.convs2.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d))))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)


class MRF(nn.Module):
    def __init__(self,
            channels,
            kernel_sizes=[3, 7, 11],
            dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for k, d in zip(kernel_sizes, dilation_rates):
            self.blocks.append(
                    ResBlock(channels, k, d))

    def forward(self, x):
        out = 0
        for block in self.blocks:
            out += block(x)
        return out

    def remove_weight_norm(self):
        for block in self.blocks:
            remove_weight_norm(block)


class Decoder(nn.Module):
    def __init__(self,
                 hubert_channels=768,
                 upsample_initial_channels=512,
                 deconv_strides=[10, 8],
                 deconv_kernel_sizes=[20, 16],
                 resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 n_fft=16,
                 hop_length=4,
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.num_kernels = len(resblock_kernel_sizes)
        self.pre = nn.Conv1d(hubert_channels, upsample_initial_channels, 7, 1, 3)
        self.f0_enc = F0Encoder(upsample_initial_channels)
        self.amp_enc = AmplitudeEncoder(upsample_initial_channels)

        self.ups = nn.ModuleList([])
        for i, (s, k) in enumerate(zip(deconv_strides, deconv_kernel_sizes)):
            self.ups.append(
                    weight_norm(
                        nn.ConvTranspose1d(
                            upsample_initial_channels//(2**i),
                            upsample_initial_channels//(2**(i+1)),
                            k, s, (k-s)//2)))

        self.MRFs = nn.ModuleList([])
        for i in range(len(self.ups)):
            c = upsample_initial_channels//(2**(i+1))
            self.MRFs.append(MRF(c, resblock_kernel_sizes, resblock_dilation_rates))

        self.post = nn.Conv1d(c, n_fft+2, 7, 1, 3)
        self.pad = nn.ReflectionPad1d([0, 1])
        self.ups.apply(init_weights)


    def mag_phase(self, x, f0, amp):
        x = self.pre(x)
        f0 = self.f0_enc(f0)
        amp = self.amp_enc(amp)
        x = x + f0 + amp
        for up, MRF in zip(self.ups, self.MRFs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = MRF(x) / self.num_kernels
        x = F.leaky_relu(x)
        x = self.post(x)
        x = self.pad(x)

        mag, phase = x.chunk(2, dim=1)
        return mag, phase


    def forward(self, x, f0, amp):
        dtype = x.dtype
        mag, phase = self.mag_phase(x, f0, amp)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        return torch.istft(s, self.n_fft, hop_length=self.hop_length)


    def remove_weight_norm(self):
        remove_weight_norm(self.pre)
        remove_weight_norm(self.post)
        for up in self.ups:
            remove_weight_norm(up)
        for MRF in self.MRFs:
            remove_weight_norm(MRF)


class DecoderONNXWrapper(nn.Module):
    def __init__(self, decoder):
        self.decoder = decoder
        super().__init__()
