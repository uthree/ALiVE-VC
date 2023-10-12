import argparse
import random

import torch

from module.voice_library import VoiceLibrary
from module.content_encoder import ContentEncoder
from module.spectrogram import spectrogram
from module.dataset import WaveFileDirectory

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate voice library from wave files")

parser.add_argument("dataset")
parser.add_argument("-lib", "--voice-library-path", default="voice_library.pt")
parser.add_argument('-cep', '--content-encoder-path', default="content_encoder.pt")

args = parser.parse_args()

ds = WaveFileDirectory(
        [args.dataset],
        length=131072,
        max_files=-1
        )

dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)


CE = ContentEncoder()
CE.load_state_dict(torch.load(args.content_encoder_path, map_location='cpu'))
VL = VoiceLibrary()

print("Generating Library...")
for i, wave in tqdm(enumerate(dl), total=512):
    n = random.randint(0, 511)
    t = CE(spectrogram(wave))[0, :, n]
    VL.tokens.data[:, :, n] = t
    if i == 512:
        break
print("Writing file...")
torch.save(VL.state_dict(), args.voice_library_path)
print("Complete!")
