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
    q, k, v: (B, H, T, Dh)
    mask: (T, T) or (B, 1, T, T) with True for allowed, False for masked
    Returns: (output, attn_weights)
    """
    Dh = q.size(-1)
    att = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)  # (B,H,T,T)
    if mask is not None:
        # mask: True=keep, False=mask
        att = att.masked_fill(~mask, float("-inf"))
    w = F.softmax(att, dim=-1)
    if dropout_p > 0.0:
        w = F.dropout(w, p=dropout_p, training=q.requires_grad)
    out = torch.matmul(w, v)  # (B,H,T,Dh)
    return out, w

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.wq(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = self.wk(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)

        # build causal mask if needed
        if mask is None and self.causal:
            # shape (1,1,T,T) broadcastable
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            mask = causal.unsqueeze(0).unsqueeze(0)
        elif mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        y, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout.p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(self.dropout(y))
