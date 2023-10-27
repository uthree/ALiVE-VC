from module.decoder import Decoder
from module.common import AdaptiveChannelNorm
import torch

decoder = Decoder()
decoder.load_state_dict(torch.load('./decoder.pt', map_location='cpu'))
decoder.last_norm = AdaptiveChannelNorm(512, 512)
torch.save(decoder.state_dict(), './decoder.pt')
