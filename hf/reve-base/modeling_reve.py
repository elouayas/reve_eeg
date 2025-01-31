import math
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel

from .configuration_reve import ReveConfig


try:
    import flash_attn

    FLASH_AVALIABLE = True
except ImportError:
    FLASH_AVALIABLE = False
    print("flash_attn not found, install it with `pip install flash_attn` if you want to use it")


#################################################################################
#                                  Layers                                       #
#################################################################################


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#################################################################################
#                                  Attention                                    #
#################################################################################


class ClassicalAttention(nn.Module):
    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads
        if self.use_sdpa:
            assert version.parse(torch.__version__) >= version.parse("2.2.0"), (
                "in order to use sdpa, you must be using pytorch 2.2 or above"
            )

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.use_sdpa:  # SDPA Implementation
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v)
        else:  # Naive Implementation
            _, _, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = nn.Softmax(dim=-1)(dots)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class FlashAttention(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = qkv.shape[:2]

        qkv = rearrange(qkv, "b n (three h d) -> (b n) three h d", three=3, h=self.num_heads)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=qkv.device)

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            seq_len,  # max seq len
            0.0,
            causal=False,
        )

        out = rearrange(out, "(b n) h d -> b n (h d)", b=batch_size)
        return out


class Attention(nn.Module):
    """
    Common API for both classical and flash attention
    """

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, use_flash: bool = True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.use_flash = use_flash
        self.attend = FlashAttention(self.heads) if use_flash else ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


#################################################################################
#                                  Transformer                                  #
#################################################################################


class TransformerBackbone(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self.dim, heads=heads, head_dim=head_dim, use_flash=FLASH_AVALIABLE),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ]
                )
            )

    def forward(self, x, return_out_layers=False) -> Union[torch.Tensor, list[torch.Tensor]]:
        out_layers = [x] if return_out_layers else None
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if return_out_layers:
                out_layers.append(x)
        return out_layers if return_out_layers else x


#################################################################################


class Learnable4DPE(nn.Module):
    def __init__(self, embed_dim: int, positions: torch.Tensor, n_timesteps: int):
        super().__init__()
        self.embed_dim = embed_dim

        assert positions.dim() == 2 and positions.size(1) == 3, "Positions should be a 2D tensor of shape (n_positions, 3)"  # noqa
        self.register_buffer("positions", positions)
        self.n_positions = len(positions)
        self.n_timesteps = n_timesteps

        self.spatial_pe = nn.Embedding(self.n_positions, self.embed_dim)
        self.temporal_pe = nn.Embedding(self.n_timesteps, self.embed_dim)

    def _convert_positions(self, pos: torch.Tensor, eps: float = 1e-5):
        """Turn a tensor of positions into a tensor of indices."""
        with torch.autocast("cuda" if pos.is_cuda else "cpu"):
            indices = torch.cdist(pos, self.positions.to(pos)).argmin(dim=-1)
            assert self.positions[indices].to(pos).allclose(pos, atol=eps), "Positions do not match"
        return indices

    def forward(self, pos, n_timesteps: int = None):
        if n_timesteps is None:
            n_timesteps = self.n_timesteps
        time_indices = torch.arange(n_timesteps)

        B, C, _ = pos.shape
        pos = self._convert_positions(pos)
        spatial_pe = self.spatial_pe(pos)  # B, C, E
        temporal_pe = self.temporal_pe(time_indices)  # T, E

        spatial_pe = spatial_pe.unsqueeze(2).expand(-1, -1, n_timesteps, -1)  # B, C, T, E
        temporal_pe = temporal_pe.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

        pe = spatial_pe + temporal_pe  # B, C, T, E

        pe = rearrange(pe, "b c t e -> b (c t) e")
        return pe


##################################################################################
#                                       4D PE                                    #
##################################################################################


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.
    """

    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z + positions[..., 3] * p_w).view(*U, -1)
        if self.dimension != 512:  # noqa
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos, num_patches):
        """
        Expand the position tensor by adding a time dimension, handling batched data.

        Args:
        - pos (Tensor): Input tensor of shape (B, C, 3), where B is the batch size,
        C is the number of channels, and 3 represents x, y, z.
        - num_patches (int): The number of time patches.

        Returns:
        - Tensor: Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
        """
        B, C, _ = pos.shape
        # Repeat each position for each time step
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)  # Shape: (B, C, num_patches, 3)
        # Generate time values with the specified increment
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()  # Shape: (num_patches,)
        time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)  # (B, C, num_patches, 1)
        # Concatenate the repeated positions with the time values along the last dimension
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)  # Shape: (B, C, num_patches, 4)
        # Reshape to (B, C * num_patches, 4)
        pos_with_time = pos_with_time.view(B, C * num_patches, 4)

        return pos_with_time


def patch_embedding(embed_dim, patch_size):
    to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
    return to_patch_embedding


def mlp_pos_embedding(embed_dim):
    mlp_pos_embedding = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
    return mlp_pos_embedding


#################################################################################
#                                       REVE                                    #
#################################################################################


class Reve(PreTrainedModel):
    config_class = ReveConfig

    def __init__(self, reve_config: ReveConfig):
        super().__init__(reve_config)

        self.embed_dim = reve_config.embed_dim
        self.freqs = reve_config.freqs
        self.patch_size = reve_config.patch_size
        self.overlap_size = reve_config.patch_overlap
        self.noise_ratio = reve_config.noise_ratio

        self.transformer = TransformerBackbone(
            dim=reve_config.embed_dim,
            depth=reve_config.depth,
            heads=reve_config.heads,
            head_dim=reve_config.head_dim,
            mlp_dim=int(reve_config.embed_dim * reve_config.mlp_dim_ratio),
            geglu=reve_config.use_geglu,
        )

        self.to_patch_embedding = patch_embedding(self.embed_dim, self.patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

        self.final_layer = nn.Identity()

        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.post_init()

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        return_output: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the model.
        Args:
            eeg (torch.Tensor): Input EEG tensor of shape (batch_size, channels, sequence_length).
            pos (torch.Tensor): Position tensor of shape (batch_size, channels, 3) representing (x, y, z) coordinates.
            return_output (bool, optional): If True, returns the output from the transformer directly.
                If False, applies the final layer and returns the processed output. Default is False.
        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: The output tensor(s) from the model. If `return_output` is True,
                returns the transformer output; otherwise, returns the output after the final layer.
        """

        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        _b, c, h, _p = patches.shape

        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        x = rearrange(self.to_patch_embedding(patches), "b c h e -> b (c h) e", c=c, h=h, e=self.embed_dim) + pos_embed
        x = self.transformer(x, return_output)
        if return_output:
            return x

        x = rearrange(x, "b (c h) e -> b c h e", b=_b, c=c, h=h, e=self.embed_dim)
        x = self.final_layer(x)
        return x

    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling on the sequence dimension of x.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, S, E), where B is the batch size,
                              C is the number of channels, S is the sequence length,
                              and E is the embedding dimension.
        Returns:
            torch.Tensor: Output tensor of shape (B, E) after attention pooling.
        """

        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")  # (B, C*S, E)
        query_output = self.cls_query_token.expand(b, -1, -1)  # (B, 1, E)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.embed_dim**0.5)  # (B, 1, C*S)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, C*S)
        out = torch.matmul(attention_weights, x).squeeze(1)  # (B, E)
        return out
