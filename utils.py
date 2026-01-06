from transformers.pytorch_utils import Conv1D
from dataclasses import dataclass
import torch.nn as nn
from transformers import AutoConfig
import torch.autograd
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

@dataclass
class quantConfig():
    W_bit: int = 4
    A_bit: int = 8
    KV_bit: int = 4
    A_layerwise: bool = False
    W_layerwise: bool = True
    KV_layerwise: bool = False
    A_quant_method: str = "symmetric"
    layerwise: bool = False
    gradclip: tuple = None  # None means no gradient clipping during training

class SymQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, layerwise: bool = False, clip_val: tuple = None):
        ctx.save_for_backward(input)
        ctx.clip_val = clip_val 
        
        if layerwise:
            max_val = torch.max(torch.abs(input))
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
                tmp = input.reshape(input.shape[0], input.shape[1], -1)
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
        X_q = torch.round(input / alpha) * alpha

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


def quantize_model(model, quant_config=quantConfig(), skip_layers=None):
    """
    Quantize model layers.
    
    Args:
        skip_layers: list of layer names to skip (e.g., ['lm_head'] to preserve output quality)
    """
    if skip_layers is None:
        skip_layers = ['lm_head']  # Don't quantize the output projection by default
    
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention):
            # print(module.state_dict())
            print((module.c_attn.weight.shape))
            # setattr(parent, child_name, GPTQuantAtttention(module, quant_config))
    
    replacements = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if any(skip in full_name for skip in skip_layers):
                print(f"Skipping quantization for: {full_name}")
                continue
            if isinstance(child, (Conv1D, nn.Linear)) and not isinstance(child, QuantLinear):
                replacements.append((module, child_name, child))
    
    for parent, child_name, child in replacements:
        setattr(parent, child_name, QuantLinear(child, quant_config))
    return model