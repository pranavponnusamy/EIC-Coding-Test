import torch
import torch.nn as nn
import math
from transformers.pytorch_utils import Conv1D
from quantutils import QuantLinear
import os

_CURRENT_PRECISION = None

@staticmethod
def set_lora_precision(precision: str):
    global _CURRENT_PRECISION
    _CURRENT_PRECISION = precision

@staticmethod
def get_active_precision():
    return _CURRENT_PRECISION

class MultiPrecisionLoRALayer(nn.Module):
    """LoRA layer with separate A/B matrices for each precision level"""
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.precisions = precisions  # e.g., ['4bit', '8bit', '16bit']
        
        # Determine dimensions
        if isinstance(layer, (Conv1D, QuantLinear)):
            in_features = layer.weight.shape[0]
            out_features = layer.weight.shape[1]
        else:
            in_features = layer.in_features
            out_features = layer.out_features
        
        # Create separate LoRA params for each precision
        self.lora_A = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(in_features, r)) for p in precisions
        })
        self.lora_B = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, out_features)) for p in precisions
        })
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.precisions:
            nn.init.kaiming_uniform_(self.lora_A[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[p])
    
    def forward(self, x):
        precision = get_active_precision()
        out = self.layer(x)
        if precision is not None and precision in self.precisions:
            lora = (x @ self.lora_A[precision]) @ self.lora_B[precision]
            out = out + self.alpha * lora
        return out


class MultiPrecisionLoRAAttentionKV(nn.Module):
    """K/V-only LoRA for attention with separate params per precision"""
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0, hidden_dim: int = 768):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.hidden_dim = hidden_dim
        self.precisions = precisions
        
        # Separate K LoRA for each precision
        self.lora_A_k = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(hidden_dim, r)) for p in precisions
        })
        self.lora_B_k = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, hidden_dim)) for p in precisions
        })
        
        # Separate V LoRA for each precision
        self.lora_A_v = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(hidden_dim, r)) for p in precisions
        })
        self.lora_B_v = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, hidden_dim)) for p in precisions
        })
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.precisions:
            nn.init.kaiming_uniform_(self.lora_A_k[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[p])
            nn.init.kaiming_uniform_(self.lora_A_v[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v[p])
    
    def forward(self, x):
        precision = get_active_precision()
        base_out = self.layer(x)
        
        if precision is not None and precision in self.precisions:
            lora_k = (x @ self.lora_A_k[precision]) @ self.lora_B_k[precision]
            lora_v = (x @ self.lora_A_v[precision]) @ self.lora_B_v[precision]
            zeros_q = torch.zeros_like(lora_k)
            lora_out = torch.cat([zeros_q, lora_k, lora_v], dim=-1)
            base_out = base_out + self.alpha * lora_out
        
        return base_out

def apply_lora(model, precisions: list, r: int = 4, alpha: float = 1.0):
    """Apply multi-precision LoRA to model"""
    hidden_dim = model.config.hidden_size
    
    # Replace c_attn in attention layers
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'GPT2Attention':
            module.c_attn = MultiPrecisionLoRAAttentionKV(
                module.c_attn, precisions, r=r, alpha=alpha, hidden_dim=hidden_dim
            )
            print(f"Applied MultiPrecision LoRA (K,V) to: {name}.c_attn")
    
    # Replace MLP layers
    replacements = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if 'lm_head' in full_name or 'attn.c_proj' in full_name or 'c_attn' in full_name:
                continue
            if isinstance(child, (Conv1D, torch.nn.Linear, QuantLinear)) and not isinstance(child, MultiPrecisionLoRALayer):
                replacements.append((module, child_name, child, full_name))
    
    for parent, child_name, child, full_name in replacements:
        setattr(parent, child_name, MultiPrecisionLoRALayer(child, precisions, r=r, alpha=alpha))
        print(f"Applied MultiPrecision LoRA to: {full_name}")
    
    # Freeze base weights, train LoRA
    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model

def save_multi_precision_lora(model, save_dir: str, precisions: list):
    """Save all precision-specific LoRA weights"""
    os.makedirs(save_dir, exist_ok=True)
    
    for precision in precisions:
        state = {}
        for name, param in model.named_parameters():
            if 'lora_' in name and precision in name:
                state[name] = param.data.clone()
        
        path = os.path.join(save_dir, f"lora_{precision}.pt")
        torch.save(state, path)
        print(f"Saved {len(state)} params for {precision} to {path}")