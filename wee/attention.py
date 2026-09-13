from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hand-rolled SDPA for teaching purposes.
    q, k, v: (B, H, T, Dh)
    mask: (T, T) or (B, 1, T, T) with True for allowed, False for masked
    Returns: (output, attn_weights)
    """
    Dh = q.size(-1)
    att = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
    if mask is not None:
        att = att.masked_fill(~mask, float("-inf"))
    w = F.softmax(att, dim=-1)
    if dropout_p > 0.0:
        w = F.dropout(w, p=dropout_p, training=q.requires_grad)
    out = torch.matmul(w, v)
    return out, w


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads: (B, n_kv_heads, T, Dh) -> (B, n_heads, T, Dh)."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA). When n_kv_heads == n_heads, this is standard MHA.
    When n_kv_heads == 1, this is Multi-Query Attention (MQA).

    Uses F.scaled_dot_product_attention for the actual computation (FlashAttention
    / memory-efficient kernels when available). KV caching is built in.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0
        assert n_heads % self.n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        apply_rope_fn=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if apply_rope_fn is not None:
            q = apply_rope_fn(q, freqs)
            k = apply_rope_fn(k, freqs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        k_expanded = _repeat_kv(k, self.n_rep)
        v_expanded = _repeat_kv(v, self.n_rep)

        is_causal = (kv_cache is None) and (T > 1)
        dp = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=is_causal, dropout_p=dp)

        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.wo(y), new_cache


MultiHeadAttention = GroupedQueryAttention
