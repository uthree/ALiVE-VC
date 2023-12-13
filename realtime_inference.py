import argparse
import pyaudio
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from module.spectrogram import spectrogram
from module.common import match_features, compute_f0
from module.decoder import Decoder
from module.content_encoder import ContentEncoder
from module.f0_estimator import F0Estimator
from module.voice_library import VoiceLibrary
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
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-f0ep', '--f0-estimator-path', default="f0_estimator.pt")
parser.add_argument('-b', '--buffersize', default=8, type=int)
parser.add_argument('-c', '--chunk', default=960, type=int)
parser.add_argument('-ic', '--inputchannels', default=1, type=int)
parser.add_argument('-oc', '--outputchannels', default=1, type=int)
parser.add_argument('-lc', '--loopbackchannels', default=1, type=int)
parser.add_argument('-f0', '--f0-rate', default=1, type=float)
parser.add_argument('-p', '--pitch', default=0, type=float)
parser.add_argument('-t', '--target', default='NONE')
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-a', '--alpha', default=0.0, type=float)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-lib', '--voice-library-path', default="NONE")
parser.add_argument('-wpe', '--world-pitch-estimation', default=False, type=bool)
parser.add_argument('-isr', '--input-sr', default=16000, type=int)
parser.add_argument('-osr', '--output-sr', default=16000, type=int)
parser.add_argument('-lsr', '--loopback-sr', default=16000, type=int)



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


PE = F0Estimator().to(device)
CE = ContentEncoder().to(device)
Dec = Decoder().to(device)
PE.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
Dec.load_state_dict(torch.load(args.decoder_path, map_location=device))

tgt = torch.zeros(1, 768, 0).to(device)

if args.target != "NONE":
    print("loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 16000)
    wf = wf / wf.abs().max()
    wf = wf[:1]
    tgt = CE(spectrogram(wf)).detach()[:, :, ::4]

if args.voice_library_path != "NONE":
    print(f"loading voice library {args.voice_library_path}")
    VL = VoiceLibrary().to(device)
    VL.load_state_dict(torch.load(args.voice_library_path, map_location=device))
    tgt = torch.cat([tgt, VL.tokens], dim=2)

print(f"Loaded {tgt.shape[2]} words.")


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

downsample_rate = 16000 / args.output_sr
internal_chunk = int(chunk * downsample_rate)
center = int(internal_chunk * buffer_size) // 2
end_of_output = center + internal_chunk // 2
begin_of_output = center - internal_chunk // 2

phi = 0

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

            # to spectrogram
            spec = spectrogram(data)
            # convert voice
            content = CE(spec)
            if args.world_pitch_estimation:
                f0 = compute_f0(data)
            else:
                f0 = PE.estimate(spec) * args.f0_rate

            pitch = 12 * torch.log2(f0 / 440) - 9 # Convert f0 to pitch
            
            pitch = pitch + args.pitch

            f0 = 440 * 2 ** ((pitch + 9) / 12) # Convert pitch to f0
            f0[torch.logical_or(f0.isnan(), f0.isinf())] = 0
            
            content = match_features(content, tgt, k=args.k, alpha=args.alpha)
            data, phi_out = Dec(content, f0=f0, phi=phi, crop=(begin_of_output, end_of_output))
            phi = phi_out[:, :, end_of_output].unsqueeze(2)

            pitch_center = f0.shape[2] // 2
            bar.set_description(desc=f"Output F0: {f0[0, 0, pitch_center]:.0f} Hz")

            # gain
            data = torchaudio.functional.gain(data, args.gain)
            # Upsample
            data = torchaudio.functional.resample(data, 16000, args.output_sr)

            data = data[0]

    data = data.cpu().numpy()
    data = (data) * 32768
    data = data
    data = data.astype(np.int16)
    center = buffer_size * chunk // 2
    s = center - chunk // 2
    e = center + chunk // 2
    data = data[s:e]
    data = data.tobytes()
    stream_output.write(data)
    if stream_loopback is not None:
        stream_loopback.write(data)
    bar.update(0)
