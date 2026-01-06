from transformers.cache_utils import StaticCache
from transformers.pytorch_utils import Conv1D
from dataclasses import dataclass
import torch.nn as nn
from transformers import AutoConfig
import torch.autograd
from _transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch
import torch.nn as nn
import math


@dataclass
class quantConfig():
    W_bit: int = 32
    A_bit: int = 32
    KV_bit: int = 8
    A_layerwise: bool = False
    W_layerwise: bool = True
    KV_layerwise: bool = False
    A_quant_method: str = "symmetric"
    layerwise: bool = False
    gradclip: tuple = None  # None means no gradient clipping during training
    name: str = None
    def __post_init__(self):
        if self.name is None:
            self.name = f"quant_{self.W_bit}_{self.A_bit}_{self.KV_bit}"

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
            # (batch, seq_len, num_heads, head_dim)
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

        return X_q.contiguous()

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
        self.in_features = layer.weight.shape[0]
        self.out_features = layer.weight.shape[1]
        
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
            return out.contiguous()
        else:
            return nn.functional.linear(act, weight, self.bias)


def quantize_model(model, quant_config=quantConfig(), quant_func=None, skip_layers=None):
    """
    Quantize model layers.
    """
    if skip_layers is None:
        skip_layers = ['lm_head']
    
    # Replace attention modules with our custom GPT2Attention that supports KV quantization
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child.__class__.__name__ == 'GPT2Attention' and not hasattr(child, 'quant_config'):
                new_attn = GPT2Attention(
                    config=child.config,
                    is_cross_attention=child.is_cross_attention,
                    layer_idx=child.layer_idx,
                    quant_config=quant_config,
                    quant_func=quant_func
                )
                # Copy weights - ensure contiguous memory layout
                new_attn.c_attn.weight.data = child.c_attn.weight.data.clone().contiguous()
                new_attn.c_attn.bias.data = child.c_attn.bias.data.clone().contiguous()
                new_attn.c_proj.weight.data = child.c_proj.weight.data.clone().contiguous()
                new_attn.c_proj.bias.data = child.c_proj.bias.data.clone().contiguous()
                if hasattr(child, 'q_attn'):
                    new_attn.q_attn.weight.data = child.q_attn.weight.data.clone().contiguous()
                    new_attn.q_attn.bias.data = child.q_attn.bias.data.clone().contiguous()
                
                setattr(module, child_name, new_attn)
                print(f"Replaced attention: {name}.{child_name}" if name else f"Replaced attention: {child_name}")
    
    # Replace linear layers with QuantLinear
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

def set_quant_precision(model, config: quantConfig):
    """Update quantization config on all QuantLinear layers"""
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.quantConfig = config
        # attention quant config
        if hasattr(module, 'quant_config') and module.quant_config is not None:
            module.quant_config = config