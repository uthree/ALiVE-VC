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
from module.f0_estimator import F0Estimator, F0EstimatorOnnxWraper
from module.content_encoder import ContentEncoder
from module.voice_library import VoiceLibrary

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outputs', default="./onnx/")
parser.add_argument('-dep', '--decoder-path', default="decoder.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")
parser.add_argument('-f0ep', '--f0-estimator-path', default="f0_estimator.pt")
parser.add_argument('-lib', '--voice-library-path', default="voice_library.pt")

args = parser.parse_args()

device = torch.device('cpu')

PE = F0Estimator().to(device)
CE = ContentEncoder().to(device)
PE.load_state_dict(torch.load(args.f0_estimator_path, map_location=device))
CE.load_state_dict(torch.load(args.content_encoder_path, map_location=device))


if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

VL = None
if args.voice_library_path != "NONE":
    print(f"loading voice library {args.voice_library_path}")
    VL = VoiceLibrary().to(device)
    VL.load_state_dict(torch.load(args.voice_library_path, map_location=device))

print("Exporting ONNX...")

print("Exporting Pitch Estimator...")
dummy_input = torch.randn(1, 641, 256)
PE = F0EstimatorOnnxWraper(PE)
torch.onnx.export(
        PE,
        dummy_input,
        os.path.join(args.outputs, "f0_estimator.onnx"),
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "barch_size", 2: "length"}
            })

print("Exporting Content Encoder...")
dummy_input = torch.randn(1, 641, 256)
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

