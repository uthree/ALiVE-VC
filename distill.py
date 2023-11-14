import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from module.spectrogram import spectrogram
from teacher.f0_estimator import F0Estimator
from student.discriminator import Discriminator
from student.model import VoiceConvertor
from module.dataset import ParallelDataset

parser = argparse.ArgumentParser(description="train student convertor")

parser.add_argument('dataset')
parser.add_argument('-dep', '--convertor-path', default="s_convertor.pt")
parser.add_argument('-disp', '--discriminator-path', default="s_discriminator.pt")
parser.add_argument('-pep', '--f0-estimator-path', default="t_f0_estimator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('--feature-matching', default=2, type=float)
parser.add_argument('--mel', default=45, type=float)
parser.add_argument('--matching', default=10, type=float)
parser.add_argument('--f0', default=10, type=float)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False

def load_or_init_models(device=torch.device('cpu')):
    dis = Discriminator().to(device)
    vc = VoiceConvertor().to(device)
    pe = F0Estimator().to(device)
    inference_mode(pe)
    if os.path.exists(args.convertor_path):
        vc.load_state_dict(torch.load(args.convertor_path, map_location=device))
    if os.path.exists(args.f0_estimator_path):
        pe.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return pe, vc, dis

def save_models(dec, dis):
    print("Saving Models...")
    torch.save(dec.state_dict(), args.convertor_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("complete!")

device = torch.device(args.device)
PE, VC, D = load_or_init_models(device)

ds = ParallelDataset(args.dataset)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptG = optim.AdamW(VC.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

SchedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(OptG, 5000)
SchedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(OptD, 5000)

mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, n_mels=80).to(device)

def cut_center(x):
    length = x.shape[2]
    center = length // 2
    size = length // 8
    return x[:, :, center-size:center+size]

def cut_center_wave(x):
    length = x.shape[1]
    center = length // 2
    size = length // 8
    return x[:, center-size:center+size]

def log_mel(x):
    return torch.log(mel(x) + 1e-6)

CEL = torch.nn.CrossEntropyLoss()

step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (src, tgt) in enumerate(dl):
        N = src.shape[0]

        amp = torch.rand(N, 1, device=device) * 2
        src = src.to(device) * amp
        tgt = tgt.to(device) * amp
        src_f0 = PE.estimate(spectrogram(src))
        tgt_f0 = PE.estimate(spectrogram(tgt))

        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            con, _ = VC.encoder(src)
            con_tgt, _ = VC.encoder(tgt)
            con_matched = VC.vector_matcher.match(con_tgt, alpha=0)

            loss_matcher = (con_matched - con).abs().mean()

            wave_fake = VC.decoder(con, tgt_f0)
            wave_recon = VC.decoder(con_tgt, tgt_f0)
            loss_adv = 0
            for logit in D.logits(cut_center_wave(wave_fake)):
                loss_adv += (logit ** 2).mean()
            for logint in D.logits(cut_center_wave(wave_recon)):
                loss_adv += (logit ** 2).mean()
            loss_mel = (log_mel(wave_fake) - log_mel(tgt)).abs().mean() + (log_mel(wave_recon) - log_mel(tgt)).abs().mean() 
            loss_feat = D.feat_loss(cut_center_wave(wave_fake), cut_center_wave(tgt)) +\
                    D.feat_loss(cut_center_wave(wave_recon), cut_center_wave(tgt)) 

            _, estimated_f0 = VC.encoder(src)
            estimated_f0 = estimated_f0.transpose(1, 2)
            src_f0 = src_f0.transpose(1, 2)
            estimated_f0 = torch.flatten(estimated_f0, 0, 1)
            src_f0 = torch.flatten(src_f0, 0, 1).squeeze(1)
            loss_f0 = CEL(estimated_f0, src_f0.to(torch.long))

            loss_g = loss_adv + loss_mel * args.mel + loss_feat * args.feature_matching + loss_f0 * args.f0 + loss_matcher * args.matching

        scaler.scale(loss_g).backward()
        scaler.step(OptG)

        # Train D.
        OptD.zero_grad()
        wave_fake = wave_fake.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = D.logits(cut_center_wave(wave_fake))
            logits_real = D.logits(cut_center_wave(tgt))
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
        
        tqdm.write(f"Step {step_count}, D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}, F0: {loss_f0.item():.4f}, V.M.: {loss_matcher.item():.4f}")

        bar.update(N)

        if batch % 300 == 0:
            save_models(VC, D)

print("Training Complete!")
save_models(VC, D)

