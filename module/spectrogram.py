import torch
import matplotlib.pyplot as plt


def spectrogram(x):
    dtype = x.dtype
    x = x.to(torch.float)
    x = torch.stft(x, 2048, 640, 2048, center=True, return_complex=True).abs()
    x = x.to(dtype)
    return x[:, :, :-1]


def plot_spectrogram(x, save_path="./spectrogram.png", log=True):
    if log:
        x = torch.log10(x ** 2 + 1e-6)
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)

