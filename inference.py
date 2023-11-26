import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from module.spectrogram import spectrogram
from module.f0_estimator import F0Estimator
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.common import match_features, compute_f0
from module.voice_library import VoiceLibrary


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-f0ep', '--f0-estimator-path', default="f0_estimator.pt")
parser.add_argument('-f0', '--f0-rate', default=1.0, type=float)
parser.add_argument('-p', '--pitch', default=0, type=float)
parser.add_argument('-int', '--intonation', default=1.0, type=float)
parser.add_argument('-t', '--target', default='NONE')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-a', '--alpha', default=0.0, type=float)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-c', '--chunk', default=48000, type=int)
parser.add_argument('-lib', '--voice-library-path', default="NONE")
parser.add_argument('-noise', '--noise-amp', default=1.0, type=float)
parser.add_argument('-harmonics', '--harmonics-amp', default=1.0, type=float)
parser.add_argument('-pf', '--post-filter-alpha', default=0.0, type=float)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False)
parser.add_argument('-norm', '--normalize', default=False, type=bool)

args = parser.parse_args()

device = torch.device(args.device)

# Plot spectrogram
def plot_spec(x, save_path="./spectrogram.png", log=False):
    plt.figure()
    x = x[0]
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)
    plt.close()


PE = F0Estimator().to(device)
CE = ContentEncoder().to(device)
Dec = Decoder().to(device)
PE.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
Dec.load_state_dict(torch.load(args.decoder_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

tgt = torch.zeros(1, 768, 0).to(device)

if args.target != "NONE":
    print("loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 48000)
    wf = wf / wf.abs().max()
    wf = wf[:1]
    tgt = CE(spectrogram(wf)).detach()

if args.voice_library_path != "NONE":
    print(f"loading voice library {args.voice_library_path}")
    VL = VoiceLibrary().to(device)
    VL.load_state_dict(torch.load(args.voice_library_path, map_location=device))
    tgt = torch.cat([tgt, VL.tokens], dim=2)

print(f"Loaded {tgt.shape[2]} words.")

paths = glob.glob(os.path.join(args.inputs, "*"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf_in = wf
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 48000)
    wf = wf / wf.abs().max()
    wf = wf.mean(dim=0, keepdim=True)
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3))], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))
    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)

    result = []
    with torch.no_grad():
        print(f"converting {path}")
        for chunk in tqdm(chunks):
            chunk = chunk.squeeze(1)

            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)
            spec = spectrogram(chunk)
            if args.world_pitch_estimation:
                f0 = compute_f0(chunk)
            else:
                f0 = PE.estimate(spec)

            # Pitch Shift and Intonation Multiply
            pitch = 12 * torch.log2(f0 / 440) - 9 # Convert f0 to pitch
            
            mean_pitch = pitch.masked_select(torch.logical_not(torch.logical_or(pitch.isinf(), pitch.isnan()))).mean()
            intonation = (pitch - mean_pitch)
            pitch = mean_pitch + intonation * args.intonation + args.pitch # Intonation Multiply

            f0 = 440 * 2 ** ((pitch + 9) / 12) # Convert pitch to f0
            f0[torch.logical_or(f0.isnan(), f0.isinf())] = 0

            feat = CE(spec)
            feat = match_features(feat, tgt, k=args.k, alpha=args.alpha)
            chunk, _, _ = Dec(feat, f0 * args.f0_rate, 0,
                        args.post_filter_alpha, args.noise_amp, args.harmonics_amp)
            
            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk.to('cpu'))
        wf = torch.cat(result, dim=1)[:, :total_length]
        wf = torchaudio.functional.resample(wf, 48000, sr)
        wf = torchaudio.functional.gain(wf, args.gain)
    wf = wf.cpu().detach()
    if args.normalize:
        wf = wf / wf.abs().max()
    file_name = f"{i}_{os.path.splitext(os.path.basename(path))[0]}"
    torchaudio.save(os.path.join(args.outputs, f"{file_name}.wav"), src=wf, sample_rate=sr)

    mel_hq = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=3840,
            hop_length=240,
            n_mels=256
            )

    def log_mel_hq(x, eps=1e-5):
        return torch.log(mel_hq(x) + eps)[:, :, 1:]

    plot_spec(log_mel_hq(wf), os.path.join(args.outputs, f"{file_name}_output_spec.png"))
    plot_spec(log_mel_hq(wf_in), os.path.join(args.outputs, f"{file_name}_input_spec.png"))
