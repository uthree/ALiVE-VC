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
from module.f0_estimator import F0Estimator
from module.decoder import Decoder
from module.voice_library import VoiceLibrary
from module.discriminator import Discriminator
from module.common import match_features, compute_f0

parser = argparse.ArgumentParser(description="train Vocoder")

parser.add_argument('dataset')
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-f0ep', '--f0-estimator-path', default="f0_estimator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=76800, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)
parser.add_argument('--feature-matching', default=2, type=float)
parser.add_argument('--mel', default=45, type=float)
parser.add_argument('--content', default=1, type=float)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False, type=bool)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False


def load_or_init_models(device=torch.device('cpu')):
    dis = Discriminator().to(device)
    ce = ContentEncoder().to(device)
    pe = F0Estimator().to(device)
    dec = Decoder().to(device)
    inference_mode(ce)
    inference_mode(pe)
    if os.path.exists(args.content_encoder_path):
        ce.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
    if os.path.exists(args.f0_estimator_path):
        pe.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return ce, pe, dec, dis


def save_models(dec, dis):
    print("Saving Models...")
    torch.save(dec.state_dict(), args.decoder_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("complete!")

def cut_center(x):
    length = x.shape[2]
    center = length // 2
    size = length // 8
    return x[:, :, center-size:center+size]

def cut_center_wav(x):
    length = x.shape[1]
    center = length // 2
    size = length // 8
    return x[:, center-size:center+size]


device = torch.device(args.device)
ce, pe, dec, D = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptG = optim.AdamW(dec.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

SchedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(OptG, 5000)
SchedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(OptD, 5000)

mel = torchaudio.transforms.MelSpectrogram(48000, n_fft=3840, hop_length=960, n_mels=192).to(device)
def log_mel(x):
    x = mel(x)
    x = torch.log(x + 1e-5)
    return x

step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device) * (torch.rand(wave.shape[0], 1, device=device) * 2)
        spec = spectrogram(wave)
        N = wave.shape[0]
        
        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                if args.world_pitch_estimation:
                    f0 = compute_f0(wave)
                else:
                    f0 = pe.estimate(spec)
                content = ce(spec)
            wave_recon, _ = dec(match_features(cut_center(content), content), cut_center(f0))
            wave_fake, _ = dec(match_features(cut_center(content), content.roll(1, dims=0)),
                                   cut_center(f0) * (1.0 + torch.rand(1, 1, device=device)))
            logits = D.logits(wave_fake) + D.logits(wave_recon)
            
            loss_mel = (log_mel(wave_recon) - log_mel(cut_center_wav(wave))).abs().mean()
            loss_feat = D.feat_loss(wave_recon, cut_center_wav(wave))
            loss_con = (cut_center(content) - ce(spectrogram(wave_recon))).abs().mean()

            loss_adv = 0
            for logit in logits:
                loss_adv += (logit ** 2).mean()
            
            loss_g = loss_mel * args.mel + loss_feat * args.feature_matching + loss_con * args.content + loss_adv
        scaler.scale(loss_g).backward()
        scaler.step(OptG)

        # Train D.
        OptD.zero_grad()
        wave_fake = wave_fake.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = D.logits(wave_fake)
            logits_real = D.logits(cut_center_wav(wave))
            loss_d = 0
            for logit in logits_real:
                loss_d += (logit ** 2).mean()
            for logit in logits_fake:
                loss_d += ((logit - 1) ** 2).mean()
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        SchedulerD.step()
        SchedulerG.step()

        step_count += 1
        
        tqdm.write(f"Step {step_count}, D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}, Con.: {loss_con.item():.4f}")

        bar.update(N)

        if batch % 300 == 0:
            save_models(dec, D)

print("Training Complete!")
save_models(dec, D)

