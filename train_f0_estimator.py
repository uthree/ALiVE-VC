import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm

from module.dataset import WaveFileDirectoryWithF0
from module.spectrogram import spectrogram
from module.f0_estimator import F0Estimator


parser = argparse.ArgumentParser(description="train f0 estimator")

parser.add_argument('dataset')
parser.add_argument('-mp', '--model-path', default="f0_estimator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=100, type=int)
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    m = F0Estimator().to(device)
    if os.path.exists(args.model_path):
        m.load_state_dict(torch.load(args.model_path, map_location=device))
    return m


def save_models(m):
    print("Saving Models...")
    torch.save(m.state_dict(), args.model_path)
    print("complete!")

device = torch.device(args.device)
model = load_or_init_models(device)

ds = WaveFileDirectoryWithF0(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate)

criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0) in enumerate(dl):
        wave = wave.to(device)
        wave = wave * ((torch.rand(wave.shape[0], 1, device=device) * 0.75) + 0.25)
        spec = spectrogram(wave)
        f0 = f0.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            f0 = torch.floor(f0).to(torch.long)
            f0 = torch.flatten(f0.squeeze(1).transpose(0, 1),
                               start_dim=0, end_dim=1)
            estimated_f0 = model(spec).transpose(1, 2)
            estimated_f0 = torch.flatten(estimated_f0, start_dim=0, end_dim=1)
            loss = criterion(
                    estimated_f0,
                    f0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        
        tqdm.write(f"loss: {loss.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 1000 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
