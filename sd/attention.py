import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch_size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, 3 * Dim) -> 3 tensors of shape (Batch_size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, H, dim / H) -> (Batch_size, H, Seq_len, dim / H)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # (Batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, dim / H) -> (Batch_size, H, Seq_len, dim / H)
        output = weight @ v

        # (Batch_size, H, Seq_len, dim / H) -> (Batch_size, Seq_len, H, dim / H)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output) 

        # (Batch_size, Seq_len, Dim)
        return output
    

class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed:int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x, y):
        # x : (latent): (Batch_size, Seq_len_Q, Dim_Q)
        # y : (context): (Batch_size, Seq_len_KV, Dim_KV) = (Batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1,2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output
         