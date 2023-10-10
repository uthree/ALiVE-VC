import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
from tqdm import tqdm

from module.spectrogram import spectrogram
from module.pitch_estimator import PitchEstimator
from module.content_encoder import ContentEncoder
from module.decoder import Decoder
from module.common import match_features
from module.voice_library import VoiceLibrary

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-pep', '--pitch-estimator-path', default="pitch_estimator.pt")
parser.add_argument('-f0', '--f0-rate', default=1.0, type=float)
parser.add_argument('-p', '--pitch', default=0, type=float)
parser.add_argument('-int', '--intonation', default=1.0, type=float)
parser.add_argument('-t', '--target', default='NONE')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-a', '--alpha', default=0.0, type=float)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-c', '--chunk', default=131072, type=int)
parser.add_argument('-lib', '--voice-library-path', default="NONE")

args = parser.parse_args()

device = torch.device(args.device)

PE = PitchEstimator().to(device)
CE = ContentEncoder().to(device)
Dec = Decoder().to(device)
PE.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
Dec.load_state_dict(torch.load(args.decoder_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

tgt = torch.zeros(1, 768, 0).to(device)

if args.target != "NONE":
    print("loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 16000)
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
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 16000)
    wf = wf / wf.abs().max()
    wf = wf[:1]
    total_length = wf.shape[1]
    chunks = torch.split(wf, args.chunk, dim=1)
    result = []
    with torch.no_grad():
        print(f"converting {path}")
        for chunk in tqdm(chunks):
            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)
            spec = spectrogram(chunk)
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
            chunk = Dec.decode(feat, f0  * args.f0_rate)
            result.append(chunk.to('cpu'))
        wf = torch.cat(result, dim=1)[:, :total_length]
        wf = torchaudio.functional.resample(wf, 16000, sr)
        wf = torchaudio.functional.gain(wf, args.gain)
    wf = wf.cpu().detach()
    wf = wf / wf.abs().max()
    torchaudio.save(os.path.join("./outputs/", f"{os.path.splitext(os.path.basename(path))[0]}.wav"), src=wf, sample_rate=sr)
