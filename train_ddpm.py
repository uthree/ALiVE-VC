import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.spectrogram import spectrogram
from module.content_encoder import ContentEncoder
from module.pitch_estimator import PitchEstimator
from module.unet import DiffusionDecoder
from module.common import match_features

parser = argparse.ArgumentParser(description="train Vocoder")

parser.add_argument('dataset')
parser.add_argument('-dep', '--decoder-path', default="ddpm.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-pep', '--pitch-estimator-path', default="pitch_estimator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=100, type=int)
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False


def load_or_init_models(device=torch.device('cpu')):
    ce = ContentEncoder().to(device)
    pe = PitchEstimator().to(device)
    dec = DiffusionDecoder().to(device)
    inference_mode(ce)
    inference_mode(pe)
    if os.path.exists(args.content_encoder_path):
        ce.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
    if os.path.exists(args.pitch_estimator_path):
        pe.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    return ce, pe, dec


def save_models(dec):
    print("Saving Models...")
    torch.save(dec.state_dict(), args.decoder_path)
    print("complete!")


device = torch.device(args.device)
ce, pe, dec = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

Opt = optim.Adam(dec.parameters(), lr=args.learning_rate)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    losses = 0
    for batch, wave in enumerate(dl):
        wave = wave.to(device) * (torch.rand(wave.shape[0], 1, device=device) * 0.75 + 0.25)
        spec = spectrogram(wave)
        
        # Train G.
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                f0 = pe.estimate(spec)
                content = ce(spec)
            condition = dec.condition_encoder(match_features(content, content), f0)
            loss = dec.ddpm.calculate_loss(wave, condition)
            if loss.isnan().any() == False:
                losses += loss.item()
        if loss.isnan().any() == False:
            scaler.scale(loss).backward()
        if batch % args.gradient_accumulation == 0 and loss.isnan().any() == False:
            scaler.step(Opt)
            Opt.zero_grad()
            scaler.update()
        
        tqdm.write(f"Loss: {loss.item():.6f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 300 == 0:
            save_models(dec)
    tqdm.write(f"Loss Mean: {losses / len(dl)}")

print("Training Complete!")
save_models(dec)
