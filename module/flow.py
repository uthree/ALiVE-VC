import torch
import torch.nn as nn

from module.common import AdaptiveConvNeXt1d


class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        x = torch.flip(x, [1])
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
            internal_channels=192,
            hidden_channels=384,
            speaker_encoding_channels=192,
            kernel_size=5,
            dilation_rate=1):
        super().__init__()

        half_channels = input_channels // 2

        self.pre = nn.Conv1d(half_channels, internal_channels, 1, 1, 0)
        self.mid_layer = AdaptiveConvNeXt1d(internal_channels, hidden_channels, speaker_encoding_channels, kernel_size)
        self.post = nn.Conv1d(internal_channels, half_channels, 1, 1, 0)

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, speaker, reverse=False):
        x0, x1 = torch.chunk(x, 2, dim=1)

        h = self.pre(x0)
        h = self.mid_layer(h, speaker)
        h = self.post(h)

        if not reverse:
            x1 = x1 + h
        else:
            x1 = x1 - h
        
        x = torch.cat([x0, x1], 1)
        return x


class Flow(nn.Module):
    def __init__(self,
            internal_channels=192,
            hidden_channels=384,
            speaker_encoding_channels=192,
            kernel_size=5,
            dilation_rate=1,
            num_flows=6,
            ):
        super().__init__()

        self.flows = nn.ModuleList([])
        for _ in range(num_flows):
            self.flows.append(
                    ResidualCouplingLayer(
                        internal_channels,
                        hidden_channels,
                        speaker_encoding_channels,
                        kernel_size,
                        dilation_rate,
                        ))

            self.flows.append(Flip())


    def forward(self, x, speaker, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, speaker, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, speaker, reverse=True)
        return x
