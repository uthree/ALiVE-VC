import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

from student.model import VoiceConvertor

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-mp', '--model-path', default='./s_convertor.pt')
parser.add_argument('-p', '--pitch', default=0, type=float)
parser.add_argument('-t', '--target', default='NONE')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-a', '--alpha', default=0.5, type=float)
parser.add_argument('-c', '--chunk', default=65536, type=int)
parser.add_argument('-norm', '--normalize', default=True, type=bool)

args = parser.parse_args()
device = torch.device(args.device)

model = VoiceConvertor().to(device)
model.load_state_dict(torch.load(args.model_path))
if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

paths = glob.glob(os.path.join(args.inputs, "*"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 16000)
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
            chunk = model.convert(chunk, args.pitch, args.alpha)
            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk.to('cpu'))
        wf = torch.cat(result, dim=1)[:, :total_length]
        wf = torchaudio.functional.resample(wf, 16000, sr)
        if args.normalize:
            wf = wf / (wf.abs().max() + 1e-8)
    wf = wf.cpu().detach()
    torchaudio.save(os.path.join("./outputs/", f"{os.path.splitext(os.path.basename(path))[0]}.wav"), src=wf, sample_rate=sr)
