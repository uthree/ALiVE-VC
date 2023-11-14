import argparse
import sys
import json
import os
import glob
import random
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from module.spectrogram import spectrogram
from teacher.f0_estimator import F0Estimator
from teacher.decoder import Decoder
from module.common import match_features, compute_f0
from module.wavlm import load_wavlm, extract_wavlm_feature
from module.dataset import WaveFileDirectory

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-t', '--targets', default="./targets/")
parser.add_argument('-o', '--outputs', default="./parallel/")
parser.add_argument('-dep', '--decoder-path', default="t_decoder.pt")
parser.add_argument('-f0ep', '--f0-estimator-path', default="t_f0_estimator.pt")
parser.add_argument('-int', '--intonation', default=1.0, type=float)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-c', '--chunk', default=65536, type=int)
parser.add_argument('-noise', '--noise-gain', default=1.0, type=float)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False)
parser.add_argument('-norm', '--normalize', default=False, type=bool)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-l', '--length', default=131072, type=int)
args = parser.parse_args()

def convert(wf, target_wave, pitch_shift=0, alpha=0, intonation=1, k=4):
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3), device=device)], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))
    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)
    
    tgt = extract_wavlm_feature(wavlm, target_wave)

    result = []
    with torch.no_grad():
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
            i = (pitch - mean_pitch)
            pitch = mean_pitch + i * intonation + pitch_shift # Intonation Multiply

            f0 = 440 * 2 ** ((pitch + 9) / 12) # Convert pitch to f0
            f0[torch.logical_or(f0.isnan(), f0.isinf())] = 0
            
            feat = extract_wavlm_feature(wavlm, chunk)
            feat = match_features(feat, tgt, k=k, alpha=alpha)
            chunk = Dec(feat, f0)
            
            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk)
        wf = torch.cat(result, dim=1)[:, :total_length]
        return wf

device = torch.device(args.device)

PE = F0Estimator().to(device)
Dec = Decoder().to(device)
PE.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
Dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
wavlm = load_wavlm(device)


ds_i = WaveFileDirectory(
        [args.inputs],
        length=args.length,
        max_files=args.max_data,
        )

dl_i = torch.utils.data.DataLoader(ds_i, batch_size=1, shuffle=True)

ds_t = WaveFileDirectory(
        [args.targets],
        length=args.length,
        max_files=args.max_data,
         )

dl_t = torch.utils.data.DataLoader(ds_t, batch_size=1, shuffle=True)

dir_s = os.path.join(args.outputs, "src")
dir_t = os.path.join(args.outputs, "tgt")

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

if not os.path.exists(dir_s):
    os.mkdir(dir_s)

if not os.path.exists(dir_t):
    os.mkdir(dir_t)

total = min(len(dl_i), len(dl_t))
bar = tqdm(total=total)
for i, (s, t) in tqdm(enumerate(zip(dl_i, dl_t))):
    t = t.to(device)
    s = s.to(device)
    alpha = random.random() * 0.5
    pitch_shift = random.randint(-12, 12)
    s_gen = convert(t, s, pitch_shift, alpha)
    torchaudio.save(os.path.join(dir_s, f"{i}.wav"), src=s_gen.to('cpu'), sample_rate=16000)
    torchaudio.save(os.path.join(dir_t, f"{i}.wav"), src=t.to('cpu'), sample_rate=16000)
    bar.update(1)
