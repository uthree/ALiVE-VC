import argparse
import pyaudio
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

from student.model import VoiceConvertor
from module.logo import print_logo

import json
import pyaudio
from tqdm import tqdm

print_logo()

parser = argparse.ArgumentParser(description="Convert voice")

audio = pyaudio.PyAudio()
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Compute device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-g', '--gain', default=0.0, type=float)
parser.add_argument('-ig', '--input-gain', default=0.0, type=float)
parser.add_argument('-mp', '--model-path', default="s_convertor.pt")
parser.add_argument('-b', '--buffersize', default=16, type=int)
parser.add_argument('-c', '--chunk', default=1024, type=int)
parser.add_argument('-ic', '--inputchannels', default=1, type=int)
parser.add_argument('-oc', '--outputchannels', default=1, type=int)
parser.add_argument('-lc', '--loopbackchannels', default=1, type=int)
parser.add_argument('-a', '--alpha', default=0.5, type=float)
parser.add_argument('-p', '--pitch', default=0, type=float)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-isr', '--input-sr', default=16000, type=int)
parser.add_argument('-osr', '--output-sr', default=16000, type=int)
parser.add_argument('-lsr', '--loopback-sr', default=16000, type=int)
parser.add_argument('-ll', '--low-latency-mode', default=True, type=bool)

args = parser.parse_args()
device_name = args.device

print(f"selected device: {device_name}")
if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

device = torch.device(device_name)
input_buff = []
chunk = args.chunk
buffer_size = args.buffersize


model = VoiceConvertor().to(device)
model.load_state_dict(torch.load(args.model_path))

stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=args.input_sr,
        channels=args.inputchannels,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=args.output_sr, 
        channels=args.outputchannels,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=args.loopback_sr, 
        channels=args.loopbackchannels,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

print("converting voice...")
print("")
bar = tqdm()

while True:
    data = stream_input.read(chunk)
    data = np.frombuffer(data, dtype=np.int16)
    input_buff.append(data)
    if len(input_buff) > buffer_size:
        del input_buff[0]
    else:
        continue

    data = np.concatenate(input_buff, 0)
    data = data.astype(np.float32) / 32768 # convert -1 to 1
    data = torch.from_numpy(data).to(device)
    data = torch.unsqueeze(data, 0)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            # Downsample
            data = torchaudio.functional.resample(data, args.input_sr, 16000)
            data = torchaudio.functional.gain(data, args.input_gain)

            data = model.convert(data, args.pitch, args.alpha)

            # gain
            data = torchaudio.functional.gain(data, args.gain)
            # Upsample
            data = torchaudio.functional.resample(data, 16000, args.output_sr)

            data = data[0]

    data = data.cpu().numpy()
    data = (data) * 32768
    data = data
    data = data.astype(np.int16)
    if args.low_latency_mode:
        center = buffer_size * chunk - chunk
    else:
        center = buffer_size * chunk // 2
    s = center - chunk // 2
    e = center + chunk // 2
    data = data[s:e]
    data = data.tobytes()
    stream_output.write(data)
    if stream_loopback is not None:
        stream_loopback.write(data)
    bar.update(0)
