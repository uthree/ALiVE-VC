import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size -1)*dilation)/2)


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


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 channels=192,
                 num_harmonics=32,
                 frame_size=960,
                 sample_rate=48000,
                 scale=0.1
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        self.to_amps = nn.Conv1d(channels, num_harmonics, 1)
    
    # x: [N, input_channels, Lf]
    def forward(self, x, f0, phi=0, crop=(0, -1)):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.frame_size # wave length

        # to amplitudes
        amps = self.to_amps(x)

        # magnitude to amplitude
        amps = torch.exp(amps)
    
        # frequency multiplyer
        mul = (torch.arange(Nh, device=f0.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)

        # Calculate formants
        formants = f0 * mul

        # Interpolate folmants
        formants = F.interpolate(formants, Lw, mode='linear')

        # Interpolate amp
        amps = F.interpolate(amps, Lw, mode='linear')

        # Generate harmonics
        dt = torch.cumsum(formants / self.sample_rate, dim=2) # numerical integrate
        dt = dt - dt[:, :, crop[0]].unsqueeze(2)
        theta = 2 * math.pi * dt + phi
        harmonics = torch.sin(theta)
        phi = torch.asin(harmonics)
    
        harmonics = harmonics * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave, phi


class IstftnetFilter(nn.Module):
    def __init__(self,
            input_channels=192,
            speaker_embedding_channels=192,
            upsample_initial_channels=256,
            deconv_strides=[10, 8],
            deconv_kernel_sizes=[20, 16],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            n_fft=48,
            hop_length=12
            ):

        super().__init__()
        self.hop_length=hop_length
        self.n_fft=n_fft

        self.num_kernels = len(resblock_kernel_sizes)
        self.pre = nn.Conv1d(input_channels, upsample_initial_channels, 7, 1, 3)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for i, (s, k) in enumerate(zip(deconv_strides, deconv_kernel_sizes)):
            self.ups.append(
                    weight_norm(
                        nn.ConvTranspose1d(
                            upsample_initial_channels//(2**i),
                            upsample_initial_channels//(2**(i+1)),
                            k, s, (k-s)//2)))
            down_conv = nn.Conv1d(
                    upsample_initial_channels//(2**(i+1)),
                    upsample_initial_channels//(2**(i)),
                    k, s, (k-s)//2)
            self.downs.insert(0, down_conv)

        self.MRFs = nn.ModuleList([])
        for i in range(len(self.ups)):
            c = upsample_initial_channels//(2**(i+1))
            self.MRFs.append(MRF(c, resblock_kernel_sizes, resblock_dilation_rates))

        self.post = nn.Conv1d(c, n_fft+2, 7, 1, 3)
        self.h_pre = nn.Conv1d(1, c, hop_length, hop_length)
        self.pad = nn.ReflectionPad1d([1, 0])
        self.ups.apply(init_weights)

    
    def forward(self, x, h):
        mag, phase = self.mag_phase(x, h)
        mag = mag.clamp_max(6.0)
        mag = torch.exp(mag)
        s = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return torch.istft(s, n_fft=self.n_fft, center=True, hop_length=self.hop_length, onesided=True)


    def mag_phase(self, x, h):
        skips = []
        h = self.h_pre(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        x = self.pre(x) + h
        for up, MRF, s in zip(self.ups, self.MRFs, reversed(skips)):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = x + s
            x = up(x)
            x = MRF(x) / self.num_kernels
        x = F.leaky_relu(x)
        x = self.post(x)
        x = self.pad(x)
        mag, phase = x.chunk(2, dim=1)
        return mag, phase


    def remove_weight_norm(self):
        remove_weight_norm(self.pre)
        remove_weight_norm(self.post)
        for up in self.ups:
            remove_weight_norm(up)
        for MRF in self.MRFs:
            remove_weight_norm(MRF)


class Decoder(nn.Module):
    def __init__(
            self,
            channels=192,
            num_harmonics=32,
            sample_rate=48000
            ):
        super().__init__()
        self.harmonic_oscillator = HarmonicOscillator(channels, num_harmonics)
        self.istftnet = IstftnetFilter()

    def forward(self, x, f0, phi=0, crop=(-1, 0)):
        harmonics, phi = self.harmonic_oscillator(x, f0, phi, crop)
        out = self.istftnet(x, harmonics)
        return out
