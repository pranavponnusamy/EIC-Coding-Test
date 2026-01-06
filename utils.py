from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D
from dataclasses import dataclass
import torch.nn as nn
from transformers import AutoConfig
import torch.autograd

@dataclass
class quantConfig():
    W_bit: int = 12
    A_bit: int = 12
    KV_bit: int = 14
    A_layerwise: bool = False
    W_layerwise: bool = True
    A_quant_method: str = "symmetric"
    layerwise: bool = False
    gradclip: tuple = (0, 0)

class SymQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, layerwise: bool = False, clip_val: tuple = None):
        ctx.save_for_backward(input)
        ctx.clip_val = clip_val 
        
        if layerwise:
            max_val = torch.max(input.max(dim=1)[0])
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_val = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_val = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError

        # quantize and dequantize the input
        # we add a small epsilon to avoid division by zero
        alpha = max_val / ((2**(num_bits - 1) - 1) + 1e-6)
        X_q = torch.round(input / (alpha + 1e-6)) * alpha

        return X_q

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clip_val = ctx.clip_val

        # clips the output (effectively STE)
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input > clip_val[1]] = 0
            grad_input[input < clip_val[0]] = 0
        return grad_input, None, None, None


class QuantLinear(nn.Module):
    def __init__(self, layer, quant_config: quantConfig = quantConfig()):
        super().__init__()
        self.weight = nn.Parameter(layer.weight.data.clone())
        self.bias = nn.Parameter(layer.bias.data.clone()) if layer.bias is not None else None
        self.quantConfig = quant_config
        self.quantFunc = SymQuantization.apply
        self.is_conv1d = isinstance(layer, Conv1D)

    def forward(self, x):

        weight = self.weight
        if self.quantConfig.W_bit and self.quantConfig.W_bit < 32:
            weight = self.quantFunc(self.weight, self.quantConfig.W_bit, 
                                       self.quantConfig.W_layerwise, self.quantConfig.gradclip)

        act = x
        if self.quantConfig.A_bit and self.quantConfig.A_bit < 32:
            act = self.quantFunc(x, self.quantConfig.A_bit, 
                                    self.quantConfig.A_layerwise, self.quantConfig.gradclip)

        if self.is_conv1d:
            out = act @ weight
            if self.bias is not None:
                out = out + self.bias
            return out
        else:
            return nn.functional.linear(act, weight, self.bias)