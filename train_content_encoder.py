import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm

from module.dataset import WaveFileDirectory
from module.spectrogram import spectrogram
from module.content_encoder import ContentEncoder
from module.hubert import load_hubert, extract_hubert_feature


parser = argparse.ArgumentParser(description="train content encoder")

parser.add_argument('dataset')
parser.add_argument('-mp', '--model-path', default="content_encoder.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    m = ContentEncoder().to(device)
    if os.path.exists(args.model_path):
        m.load_state_dict(torch.load(args.model_path, map_location=device))
    return m


def save_models(m):
    print("Saving Models...")
    torch.save(m.state_dict(), args.model_path)
    print("complete!")

device = torch.device(args.device)
model = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate)

hubert = load_hubert(device)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device)
        spec = spectrogram(wave)
        
        # Train G.
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave_16k = torchaudio.functional.resample(wave, 44100, 16000)
            hubert_feature = extract_hubert_feature(hubert, wave_16k)
            output = model(spec)
            hubert_feature = F.interpolate(hubert_feature, output.shape[2], mode='linear')
            loss = (output - hubert_feature).abs().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        
        tqdm.write(f"loss: {loss.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 100 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
