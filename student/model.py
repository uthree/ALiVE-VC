import torch
import torch.nn as nn

from student.decoder import Decoder
from student.encoder import Encoder
from student.vector_matcher import VectorMatcher

class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()
        self.vector_matcher = VectorMatcher()

    def convert(self, wave, pitch_shift=0, alpha=0.2):
        con, f0 = self.encoder(wave)
        con = self.vector_matcher.match(con, k=4, alpha=alpha)
        f0 = f0.argmax(dim=1).to(torch.float).unsqueeze(1)

        # Pitch Shift
        pitch = 12 * torch.log2(f0 / 440) - 9 # Convert f0 to pitch
        pitch = pitch  + pitch_shift

        f0 = 440 * 2 ** ((pitch + 9) / 12) # Convert pitch to f0
        f0[torch.logical_or(f0.isnan(), f0.isinf())] = 0
        
        out = self.decoder(con, f0)
        return out

