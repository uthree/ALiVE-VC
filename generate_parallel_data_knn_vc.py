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

from module.dataset import WaveFileDirectory

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-t', '--targets', default="./targets/")
parser.add_argument('-o', '--outputs', default="./parallel/")
parser.add_argument('-int', '--intonation', default=1.0, type=float)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-c', '--chunk', default=65536, type=int)
parser.add_argument('-noise', '--noise-gain', default=1.0, type=float)
parser.add_argument('-wpe', '--world-pitch-estimation', default=False)
parser.add_argument('-norm', '--normalize', default=False, type=bool)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-l', '--length', default=65536, type=int)
args = parser.parse_args()

def convert(wf, target_wave, k=4):
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3), device=device)], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))
    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)
    
    matching_set = knn_vc.get_features(target_wave.to(device).squeeze(0))

    result = []
    with torch.no_grad():
        for chunk in tqdm(chunks):
            chunk = chunk.squeeze(1)

            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)
            query_seq = knn_vc.get_features(chunk.to(device).squeeze(0), vad_trigger_level=0)
            chunk = knn_vc.match(query_seq, matching_set, topk=k).unsqueeze(0)
            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk)
        wf = torch.cat(result, dim=1)[:, :total_length]
        return wf

knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)

device = torch.device(args.device)

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
for i, (s, t) in enumerate(zip(dl_i, dl_t)):
    t = t.to(device)
    s = s.to(device)
    alpha = random.random() * 0.5
    pitch_shift = random.randint(-12, 12)
    s_gen = convert(t, s)
    torchaudio.save(os.path.join(dir_s, f"{i}.wav"), src=s_gen.to('cpu'), sample_rate=16000)
    torchaudio.save(os.path.join(dir_t, f"{i}.wav"), src=t.to('cpu'), sample_rate=16000)
    bar.update(1)
