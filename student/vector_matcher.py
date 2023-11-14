import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorMatcher(nn.Module):
    def __init__(self, num_tokens=512, dim=768):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, dim, num_tokens))
        self.dim = dim

    def forward(self, source):
        return self.match(source)

    def match(self, source, k=4, alpha=0.5):
        reference = self.tokens.expand(
                source.shape[0],
                self.dim,
                self.tokens.shape[2])

        input_data = source

        # source: [N, 768, Length], reference: [N, 768, Length]
        source = source.transpose(1, 2)
        reference = reference.transpose(1, 2)
        source_norm = torch.norm(source, dim=2, keepdim=True)
        reference_norm = torch.norm(reference, dim=2, keepdim=True)
        cos_sims = torch.bmm((source / source_norm), (reference / reference_norm).transpose(1, 2))
        best = torch.topk(cos_sims, k, dim=2)

        result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
        result = result.transpose(1, 2)
        return result * (1-alpha) + input_data * alpha

