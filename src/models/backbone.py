"""Transformer backbone layers used by REVE encoder/decoder modules."""

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None  # type: ignore[assignment]
    FLASH_AVAILABLE = False
    print("flash_attn not found, install it with `pip install flash_attn` if you want to use it")
else:
    FLASH_AVAILABLE = True

#################################################################################
#                                  Layers                                       #
#################################################################################


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, geglu):
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
    def __init__(self, heads, use_sdpa=True):
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

    def forward_attn(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        _, _, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5
        dots = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = nn.Softmax(dim=-1)(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")

        return out, attn


class FlashAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        if flash_attn is None:
            raise RuntimeError("flash_attn is not available but FlashAttention was instantiated")
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

    def __init__(self, dim, heads=8, head_dim=64, use_flash=True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.use_flash = use_flash
        self.attend: nn.Module = self._get_attend()

    def _get_attend(self) -> nn.Module:
        if self.use_flash:
            return FlashAttention(self.heads)
        else:
            return ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)

    def forward_attn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.attend = ClassicalAttention(self.heads, use_sdpa=False)  # use classical attention for visualization

        x = self.norm(x)
        qkv = self.to_qkv(x)
        out, attn = self.attend.forward_attn(qkv)
        out = self.to_out(out)

        self.attend = self._get_attend()
        return out, attn


#################################################################################
#                                  Transformer                                  #
#################################################################################


class TransformerBackbone(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):  # noqa: PLR0913
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self.dim, heads=heads, head_dim=head_dim, use_flash=FLASH_AVAILABLE),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ],
                ),
            )

    def forward(self, x: torch.Tensor, return_out_layers: bool = False) -> list[torch.Tensor] | torch.Tensor:
        out_layers: list[torch.Tensor] = [x] if return_out_layers else []
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x
            if return_out_layers:
                out_layers.append(x)
        return out_layers if return_out_layers else x

    def forward_attn(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn: list[torch.Tensor] = []

        for attn_layer, ff_layer in self.layers:  # type: ignore
            temp, attn_ = attn_layer.forward_attn(x)
            x = temp + x
            attn.append(attn_)
            x = ff_layer(x) + x
        return x, attn


#################################################################################


def get_backbone(backbone_args) -> TransformerBackbone:
    return TransformerBackbone(
        dim=backbone_args.embed_dim,
        depth=backbone_args.depth,
        heads=backbone_args.heads,
        head_dim=backbone_args.head_dim,
        mlp_dim=int(backbone_args.embed_dim * backbone_args.mlp_dim_ratio),
        geglu=backbone_args.use_geglu,
    )
