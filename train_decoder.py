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
from module.decoder import Decoder
from module.discriminator import Discriminator
from module.common import match_features

parser = argparse.ArgumentParser(description="train Vocoder")

parser.add_argument('dataset')
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-pep', '--pitch-estimator-path', default="pitch_estimator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float)
parser.add_argument('-len', '--length', default=16384, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)
parser.add_argument('--feature-matching', default=2, type=float)
parser.add_argument('--mel', default=45, type=float)
parser.add_argument('--content', default=1, type=float)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False


def load_or_init_models(device=torch.device('cpu')):
    dis = Discriminator().to(device)
    ce = ContentEncoder().to(device)
    pe = PitchEstimator().to(device)
    dec = Decoder().to(device)
    inference_mode(ce)
    inference_mode(pe)
    if os.path.exists(args.content_encoder_path):
        ce.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
    if os.path.exists(args.pitch_estimator_path):
        pe.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
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
    size = length // 4
    return x[:, :, center-size:center+size]

def cut_center_wav(x):
    length = x.shape[1]
    center = length // 2
    size = length // 4
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

OptG = optim.AdamW(dec.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

SchedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(OptG, 5000)
SchedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(OptD, 5000)

mel = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=80).to(device)

def log_mel(x):
    return torch.log(mel(x) + 1e-6)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device) * (torch.rand(wave.shape[0], 1, device=device) * 1.5 + 0.25)
        spec = spectrogram(wave)
        
        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                f0 = pe.estimate(spec)
                content = ce(spec)
            wave_recon, mu, sigma = dec(match_features(cut_center(content), content), cut_center(f0))
            wave_fake = dec.decode(match_features(cut_center(content), content.roll(1, dims=0)),
                                   cut_center(f0) * (0.5 + 1.5 * torch.rand(1, 1, device=device)))
            logits = D.logits(wave_fake) + D.logits(wave_recon)
            
            loss_mel = (log_mel(wave_recon) - log_mel(cut_center_wav(wave))).abs().mean()
            loss_feat = D.feat_loss(wave_recon, cut_center_wav(wave))
            loss_kl = (-1 - sigma + torch.exp(sigma)).mean() + (mu ** 2).mean()
            loss_con = (cut_center(content) - ce(spectrogram(wave_recon))).abs().mean()

            loss_adv = 0
            for logit in logits:
                loss_adv += F.relu(1 - logit).mean() / len(logits)
            
            loss_g = loss_mel * args.mel + loss_feat * args.feature_matching + loss_con * args.content + loss_adv + loss_kl
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
                loss_d += F.relu(1 - logit).mean() / len(logits)
            for logit in logits_fake:
                loss_d += F.relu(1 + logit).mean() / len(logits)
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        SchedulerD.step()
        SchedulerG.step()
        
        tqdm.write(f"D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}, Con.: {loss_con.item():.4f}, K.L.: {loss_kl.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 300 == 0:
            save_models(dec, D)

print("Training Complete!")
save_models(dec, D)
