import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.common import AdaptiveConvNeXt1d, AdaptiveChannelNorm, ChannelNorm
from module.ddpm import DDPM


LRELU_SLOPE = 0.1


class ConditionEncoder(nn.Module):
    def __init__(self, channels=512, condition_channels=768):
        super().__init__()
        self.c1 = nn.Conv1d(condition_channels, channels*2, 1)
        self.c2 = nn.Conv1d(1, channels*2, 1)
        self.c3 = nn.Conv1d(channels*2, channels, 1)
        self.c2.weight.data.normal_(0, 0.3)

    def forward(self, c, f0):
        c = self.c1(c) + torch.sin(self.c2(f0))
        c = F.gelu(c)
        c = self.c3(c)
        return c


class TimeEncoding1d(nn.Module):
    def __init__(self, channels=512, max_timesteps=10000, return_encoding_only=False):
        super().__init__()
        self.channels = channels
        self.max_timesteps = max_timesteps
        self.return_encoding_only = return_encoding_only

    # t: [batch_size]
    def forward(self, x, t):
        emb = t.unsqueeze(1).expand(t.shape[0], self.channels).unsqueeze(-1)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 1 / (self.max_timesteps ** (torch.arange(self.channels//2, device=x.device) / (self.channels//2)))
        factors = factors.unsqueeze(0).unsqueeze(2)
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1).expand(*x.shape)

        ret = emb if self.return_encoding_only else x + emb
        return ret


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size -1)*dilation)/2)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])

        for d in dilation:
            self.convs1.append(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d)))
            self.convs2.append(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                        padding=get_padding(kernel_size, d)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


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
        mu = out.mean(dim=2, keepdim=True)
        sigma = out.std(dim=2, keepdim=True)
        return out


class Decoder(nn.Module):
    def __init__(self,
            input_channels=512,
            upsample_initial_channels=256,
            speaker_embedding_channels=128,
            deconv_strides=[8, 8, 4],
            deconv_kernel_sizes=[16, 16, 8],
            resblock_kernel_sizes=[3, 5, 7],
            resblock_dilation_rates=[[1, 2], [2, 6], [3, 12]]
            ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.pre = nn.Conv1d(input_channels, upsample_initial_channels, 7, 1, 3)

        self.ups = nn.ModuleList([])
        for i, (s, k) in enumerate(zip(deconv_strides, deconv_kernel_sizes)):
            self.ups.append(
                    nn.ConvTranspose1d(
                        upsample_initial_channels//(2**i),
                        upsample_initial_channels//(2**(i+1)),
                        k, s, (k-s)//2))

        self.MRFs = nn.ModuleList([])
        for i in range(len(self.ups)):
            c = upsample_initial_channels//(2**(i+1))
            self.MRFs.append(MRF(c, resblock_kernel_sizes, resblock_dilation_rates))
        
        self.post = nn.Conv1d(c, 1, 7, 1, 3)
    
    def forward(self, x):
        x = self.pre(x)
        for up, MRF in zip(self.ups, self.MRFs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = MRF(x)
        x = F.leaky_relu(x)
        x = self.post(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
            output_channels=512,
            downsample_initial_channels=16,
            speaker_embedding_channels=128,
            conv_strides=[4, 8, 8],
            conv_kernel_sizes=[8, 16, 16],
            resblock_kernel_sizes=[3, 5, 7],
            resblock_dilation_rates=[[1, 2], [2, 6], [3, 12]]
            ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.pre = nn.Conv1d(1, downsample_initial_channels, 7, 1, 3)

        self.downs = nn.ModuleList([])
        for i, (s, k) in enumerate(zip(conv_strides, conv_kernel_sizes)):
            self.downs.append(
                    nn.Conv1d(
                        downsample_initial_channels * (2**i),
                        downsample_initial_channels * (2**(i+1)),
                        k, s, (k-s)//2))

        self.MRFs = nn.ModuleList([])
        for i in range(len(self.downs)):
            c = downsample_initial_channels * (2**(i+1))
            self.MRFs.append(MRF(c, resblock_kernel_sizes, resblock_dilation_rates))
        
        self.post = nn.Conv1d(c, output_channels, 7, 1, 3)
    
    def forward(self, x):
        x = self.pre(x)
        for down, MRF in zip(self.downs, self.MRFs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = down(x)
            x = MRF(x)
        x = F.leaky_relu(x)
        x = self.post(x)
        return x


# U-Net
class UNet(nn.Module):
    def __init__(self,
                 internal_channels=512,
                 hubert_channels=768,
                 hidden_channels=1536,
                 num_layers=16,
                 ):
        super().__init__()
        self.encoder = Encoder(internal_channels)
        self.decoder = Decoder(internal_channels)
        self.last_norm = AdaptiveChannelNorm(internal_channels, 512)
        self.time_conv = nn.Conv1d(internal_channels, internal_channels, 1)
        self.mid_layers = nn.ModuleList([
            AdaptiveConvNeXt1d(internal_channels, hidden_channels, 512, scale=1/num_layers)
            for _ in range(num_layers)
            ])
        self.time_enc = TimeEncoding1d(return_encoding_only=True)


    def forward(self, x, condition, time):
        x = x.unsqueeze(1)
        skip = x
        x = self.encoder(x)
        time_emb = self.time_conv(self.time_enc(x, time))
        for l in self.mid_layers:
            x = x + time_emb
            x = l(x, condition)
        x = self.last_norm(x, condition)
        x = self.decoder(x)
        x = x + skip
        x = x.squeeze(1)
        return x


class DiffusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_encoder = ConditionEncoder()
        self.ddpm = DDPM(UNet())
