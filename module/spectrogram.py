import torch
import matplotlib.pyplot as plt


global hann_window
hann_window = {}

def spectrogram(x):
    if str(x.device) not in hann_window:
        hann_window[str(x.device)] = torch.hann_window(1024).to(x.device)
    w = hann_window[str(x.device)]
    x = torch.stft(x, n_fft=1024, hop_length=256, center=True,
                   return_complex=True, normalized=False, window=w).abs()
    return x[:, :, 1:]


def plot_spectrogram(x, save_path="./spectrogram.png", log=True):
    if log:
        x = torch.log10(x ** 2 + 1e-6)
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)

