from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import GPT, GPTBlock, GPTConfig
from .attention import MultiHeadAttention

class CachableMultiHeadAttention(MultiHeadAttention):
    """
    Drop-in replacement for wee.attention.MultiHeadAttention with KV caching.
    - forward(x, past_kv=None, use_cache=False) -> y, present_kv
    where past_kv/present_kv are tuples (k, v) with shapes (B,H,T,Dh).
    """
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.wq(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.wk(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)  # concat on time
            v = torch.cat([pv, v], dim=2)

        # build causal mask if needed
        if mask is None and self.causal:
            TT = k.size(-2)  # total time after concat
            qT = q.size(-2)
            causal = torch.tril(torch.ones(qT, TT, dtype=torch.bool, device=x.device))
            mask = causal.unsqueeze(0).unsqueeze(0)
        elif mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        from .attention import scaled_dot_product_attention
        y, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout.p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(self.dropout(y))
        present = (k, v) if use_cache else None
        return y, present

def upgrade_gpt_for_kv(model: GPT) -> GPT:
    """
    In-place swap of attention modules to CachableMultiHeadAttention, preserving weights.
    """
    for blk in model.blocks:
        old = blk.attn
        new = CachableMultiHeadAttention(old.d_model, old.n_heads, dropout=old.dropout.p, causal=old.causal)
        new.load_state_dict(old.state_dict())
        blk.attn = new
    return model

@torch.no_grad()
def generate_with_cache(model: GPT, idx: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    """
    Generate tokens using KV cache after the first step.
    """
    device = next(model.parameters()).device
    idx = idx.to(device)
    # First step: run the full prefix once to get initial cache
    B, T = idx.shape
    pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    x = model.drop(x)
    past_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for blk in model.blocks:
        # layer norm then attention with cache
        x_ln = blk.ln1(x)
        y, present = blk.attn(x_ln, past_kv=None, use_cache=True)
        past_kv.append(present)
        x = x + y
        x = x + blk.mlp(blk.ln2(x))
    x = model.ln_f(x)
    logits = model.head(x)
    # Sample first new token
    last = logits[:, -1, :] / max(temperature, 1e-6)
    if top_k is not None:
        v, _ = torch.topk(last, top_k)
        last[last < v[:, [-1]]] = -float("Inf")
    probs = F.softmax(last, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)

    # Subsequent tokens: use cache, feed only last token
    for _ in range(max_new_tokens - 1):
        pos = torch.tensor([[idx.size(1)-1]], device=device)
        x = model.tok_emb(idx[:, -1:]) + model.pos_emb(pos)
        x = model.drop(x)
        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk, pkv in zip(model.blocks, past_kv):
            x_ln = blk.ln1(x)
            y, present = blk.attn(x_ln, past_kv=pkv, use_cache=True)
            new_past.append(present)
            x = x + y
            x = x + blk.mlp(blk.ln2(x))
        past_kv = new_past
        x = model.ln_f(x)
        logits = model.head(x)
        last = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(last, top_k)
            last[last < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

def save_kv(past_kv: List[Tuple[torch.Tensor, torch.Tensor]], path: str):
    torch.save([ (k.cpu(), v.cpu()) for (k,v) in past_kv ], path)

def load_kv(path: str, device: Optional[torch.device] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data = torch.load(path, map_location=device or "cpu")
    return [ (k.to(device or "cpu"), v.to(device or "cpu")) for (k,v) in data ]
