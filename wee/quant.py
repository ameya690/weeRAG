from __future__ import annotations
from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn

def _sizeof(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

class QuantLinear(nn.Module):
    """
    Educational post-training quantized Linear.
    - Weights stored as int8 or int4 with per-tensor/per-channel scales.
    - On forward, dequantize weights to float32 and run matmul (clear, not fast).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 8, per_channel: bool = True):
        super().__init__()
        assert bits in (8, 4), "Only 8 or 4 bits supported"
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.per_channel = per_channel

        if bits == 8:
            self.register_buffer("w_q", torch.empty((out_features, in_features), dtype=torch.int8))
            self.register_buffer("scale", torch.empty((out_features, 1) if per_channel else (1, 1), dtype=torch.float32))
            self.register_buffer("zero", torch.zeros_like(self.scale))
        else:
            # store two int4 in one int8 slot; pack/unpack
            self.register_buffer("w_q", torch.empty((out_features, (in_features + 1)//2), dtype=torch.int8))
            self.register_buffer("scale", torch.empty((out_features, 1) if per_channel else (1, 1), dtype=torch.float32))
            self.register_buffer("zero", torch.zeros_like(self.scale))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def from_linear(lin: nn.Linear, bits: int = 8, per_channel: bool = True) -> "QuantLinear":
        qlin = QuantLinear(lin.in_features, lin.out_features, bias=(lin.bias is not None), bits=bits, per_channel=per_channel)
        qlin.quantize_(lin.weight.data, lin.bias.data if lin.bias is not None else None)
        return qlin

    def _calc_scale_zero(self, w: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            wmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            wmax = w.abs().amax().reshape(1,1).clamp(min=1e-8)
        qmax = (2**(self.bits-1) - 1)
        scale = wmax / qmax
        zero = torch.zeros_like(scale)
        return scale, zero

    def quantize_(self, w_fp32: torch.Tensor, b_fp32: Optional[torch.Tensor] = None):
        scale, zero = self._calc_scale_zero(w_fp32)
        self.scale.copy_(scale); self.zero.copy_(zero)
        if self.bits == 8:
            w_q = torch.clamp((w_fp32/scale).round(), -128, 127).to(torch.int8)
            self.w_q.copy_(w_q)
        else:
            # int4 pack: values in [-8,7]
            w_q = torch.clamp((w_fp32/scale).round(), -8, 7).to(torch.int8)
            # pack two 4-bit values per byte: low nibble and high nibble
            out, inf = w_q.shape
            packed = torch.empty((out, (inf+1)//2), dtype=torch.int8, device=w_q.device)
            for i in range(0, inf, 2):
                a = w_q[:, i]
                b = w_q[:, i+1] if i+1 < inf else torch.zeros_like(a)
                # convert signed [-8,7] to unsigned [0,15] by +8
                a_u = (a + 8).to(torch.uint8)
                b_u = (b + 8).to(torch.uint8)
                packed[:, i//2] = (a_u & 0x0F) | ((b_u & 0x0F) << 4)
            self.w_q.copy_(packed)
        if b_fp32 is not None and self.bias is not None:
            with torch.no_grad():
                self.bias.copy_(b_fp32)

    def _dequant_weight(self) -> torch.Tensor:
        if self.bits == 8:
            return (self.w_q.float() * self.scale)
        # int4 unpack
        out, halfcols = self.w_q.shape
        inf = self.in_features
        unpacked = torch.empty((out, inf), dtype=torch.int8, device=self.w_q.device)
        for i in range(halfcols):
            byte = self.w_q[:, i].to(torch.uint8)
            low = (byte & 0x0F).to(torch.int8) - 8
            high = ((byte >> 4) & 0x0F).to(torch.int8) - 8
            j = 2*i
            unpacked[:, j] = low
            if j+1 < inf:
                unpacked[:, j+1] = high
        return unpacked.float() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight()
        y = x.matmul(w.t())
        if self.bias is not None:
            y = y + self.bias
        return y

def quantize_model(model: nn.Module, bits: int = 8, per_channel: bool = True) -> nn.Module:
    """
    Replace nn.Linear with QuantLinear in-place (returns model for chaining).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            q = QuantLinear.from_linear(module, bits=bits, per_channel=per_channel)
            setattr(model, name, q)
        else:
            quantize_model(module, bits=bits, per_channel=per_channel)
    return model

def size_report(model: nn.Module) -> Dict[str, int]:
    """
    Report sizes in bytes of trainable parameters, non-trainable buffers, and total.
    """
    params = sum(p.numel() for p in model.parameters())
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameters": params,
        "param_bytes": params_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": params_bytes + buffer_bytes,
    }


def eval_perplexity(model: nn.Module, data_ids: torch.Tensor, vocab_size: int, chunk_len: int = 64) -> float:
    """
    Compute token-level perplexity over data_ids (shape [N]) by sliding windows.
    Ensures x and y have identical lengths on every step.
    """
    model.eval()
    import torch.nn.functional as F

    N = data_ids.numel()
    if N < 2:
        return float("inf")  # not enough tokens

    losses = []
    with torch.no_grad():
        # Iterate over prefix positions; each step uses T tokens and predicts the next T.
        for i in range(0, N - 1, chunk_len):
            T = min(chunk_len, (N - 1) - i)  # ensure we have T pairs
            x = data_ids[i : i + T].unsqueeze(0)           # (1, T)
            y = data_ids[i + 1 : i + 1 + T].unsqueeze(0)   # (1, T)
            logits, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = sum(losses) / max(len(losses), 1)
    return math.exp(mean_loss)

