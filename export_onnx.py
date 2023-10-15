import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

from module.spectrogram import spectrogram
from module.pitch_estimator import PitchEstimator
from module.content_encoder import ContentEncoder
from module.decoder import Decoder, DecoderOnnxWrapper
from module.common import match_features, compute_f0, compute_amplitude
from module.voice_library import VoiceLibrary

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outputs', default="./onnx/")
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-pep', '--pitch-estimator-path', default="pitch_estimator.pt")
parser.add_argument('-lib', '--voice-library-path', default="NONE")

args = parser.parse_args()

device = torch.device('cpu')

PE = PitchEstimator().to(device)
CE = ContentEncoder().to(device)
Dec = Decoder().to(device)
PE.load_state_dict(torch.load(args.pitch_estimator_path, map_location=device))
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))
Dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
Dec = DecoderOnnxWrapper(Dec)

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

VL = None
if args.voice_library_path != "NONE":
    print(f"loading voice library {args.voice_library_path}")
    VL = VoiceLibrary().to(device)
    VL.load_state_dict(torch.load(args.voice_library_path, map_location=device))
    tgt = torch.cat([tgt, VL.tokens], dim=2)


print("Exporting ONNX...")

print("Exporting Pitch Estimator...")
dummy_input = torch.randn(1, 513, 256)
torch.onnx.export(
        PE,
        dummy_input,
        os.path.join(args.outputs, "pitch_estimator.onnx"),
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "barch_size", 2: "length"}
            })

print("Exporting Content Encoder...")
dummy_input = torch.randn(1, 513, 256)
torch.onnx.export(
        CE,
        dummy_input,
        os.path.join(args.outputs, "content_encoder.onnx"),
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "barch_size", 2: "length"}
            })

print("Exporting Decoder...")
dummy_input = torch.randn(1, 768, 256)
dummy_f0 = torch.randn(1, 1, 256)
dummy_amp = torch.randn(1, 1, 256)
noise = torch.randn(1, 512, 256)
torch.onnx.export(
        Dec,
        (dummy_input, dummy_f0, dummy_amp, noise),
        os.path.join(args.outputs, "decoder.onnx"),
        opset_version=15,
        input_names=["input", "f0", "amplitude", "noise"],
        output_names=["magnitude", "phase"],
        dynamic_axes={
            "input": {0: "barch_size", 2: "length"},
            "f0" : {0: "batch_size", 2: "length"},
            "amplitude": {0: "batch_size", 2: "length"}
            })


print("Complete!")
