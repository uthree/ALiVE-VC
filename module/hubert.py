import torch
import torch.nn.functional as F
from transformers import WavLMModel

def load_hubert(device=torch.device('cpu')):
    print("Loading WavLM...")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_hubert_feature(wavlm, wave, segment_size=256):
    length = wave.shape[1] // segment_size
    hidden_states = wavlm(wave, output_hidden_states=True).hidden_states
    feature = hidden_states[4]
    feature = feature.transpose(1, 2)
    feature = F.interpolate(feature, length, mode='linear')
    return feature
