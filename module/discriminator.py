import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class PeriodicDiscriminator(nn.Module):
    def __init__(self,
                 channels=32,
                 period=2,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 groups = [],
                 max_channels=1024
                 ):
        super().__init__()
        self.input_layer = weight_norm(
                nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=get_padding(kernel_size, 1)))
        self.layers = nn.Sequential()
        for i in range(num_stages):
            c = min(channels * (4 ** i), max_channels)
            c_next = min(channels * (4 ** (i+1)), max_channels)
            if i == (num_stages - 1):
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
            else:
                self.layers.append(
                        weight_norm(
                            nn.Conv2d(c, c_next, (kernel_size, 1), (stride, 1), groups=groups[i],
                                      padding=get_padding(kernel_size, 1))))
                self.layers.append(
                        nn.LeakyReLU(LRELU_SLOPE))
        c = min(channels * (4 ** (num_stages-1)), max_channels)
        self.final_conv = weight_norm(
                nn.Conv2d(c, c, (5, 1), 1, padding=get_padding(5, 1)))
        self.final_relu = nn.LeakyReLU(LRELU_SLOPE)
        self.output_layer = weight_norm(
                nn.Conv2d(c, 1, (3, 1), 1, padding=get_padding(3, 1)))
        self.period = period

    def forward(self, x, logit=True):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        if logit:
            x = self.output_layer(x)
        return x

    def feat(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            if "Conv" in type(layer).__name__:
                feats.append(x)
        return feats


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 groups=[1, 4, 16, 16, 16],
                 channels=64,
                 kernel_size=5,
                 stride=3,
                 num_stages=4,
                 ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])

        for p in periods:
            self.sub_discriminators.append(
                    PeriodicDiscriminator(channels,
                                          p,
                                          kernel_size,
                                          stride,
                                          num_stages,
                                          groups=groups))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits
    
    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats


class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segment_size=16,
            channels=[64, 64, 64],
            norm_type='spectral',
            kernel_size=11,
            strides=[1, 1, 1],
            dropout_rate=0.1,
            groups=[],
            pool = 1
            ):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(pool)
        self.segment_size = segment_size
        if norm_type == 'weight':
            norm_f = nn.utils.weight_norm
        elif norm_type == 'spectral':
            norm_f = nn.utils.spectral_norm
        else:
            raise f"Normalizing type {norm_type} is not supported."
        self.layers = nn.Sequential()
        self.input_layer = norm_f(nn.Conv1d(segment_size, channels[0], 1, 1, 0))
        for i in range(len(channels)-1):
            if i == 0:
                k = 15
            else:
                k = kernel_size
            self.layers.append(
                    norm_f(
                        nn.Conv1d(channels[i], channels[i+1], k, strides[i], 0, groups=groups[i])))
            self.layers.append(
                    nn.Dropout(dropout_rate))
            self.layers.append(nn.LeakyReLU(LRELU_SLOPE))
        self.output_layer = norm_f(nn.Conv1d(channels[-1], 1, 1, 1, 0))

    def forward(self, x, logit=True):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        x = self.layers(x)
        if logit:
            x = self.output_layer(x)
        return x

    def feat(self, x):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segments=[1, 1, 1],
            channels=[64, 128, 256, 256, 256],
            kernel_sizes=[15, 41, 41, 41, 41],
            strides=[1, 2, 4, 4, 4, 4],
            groups=[1, 2, 4, 4, 4],
            pools=[1, 2, 4]
            ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for i, (k, sg, p) in enumerate(zip(kernel_sizes, segments, pools)):
            if i == 0:
                n = 'spectral'
            else:
                n = 'weight'
            self.sub_discriminators.append(
                    ScaleDiscriminator(sg, channels, n, k, strides, groups=groups, pool=p))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits

    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MSD = MultiScaleDiscriminator()
    
    def logits(self, x):
        return self.MPD(x) + self.MSD(x)
    
    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.MPD.feat(real) + self.MSD.feat(real)
        fake_feat = self.MPD.feat(fake) + self.MSD.feat(fake)
        loss = 0
        for r, f in zip(real_feat, fake_feat):
            loss = loss + F.l1_loss(f, r)
        return loss
