"""Downstream classifier heads and wrappers built on top of REVE encoder outputs."""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from models.backbone import RMSNorm
from models.encoder import REVE
from utils.initialization import ConfigInit, init_cls


H_DIM_MAP = {
    "cbramod": 200,
    "biot": 256,
    "labram": 1000,
}


class ReveClassifier(nn.Module):
    def __init__(
        self,
        encoder: REVE,
        n_classes,
        dropout,
        pooling="last",
        **kwargs,
    ):
        super().__init__()
        assert pooling in ["last", "last_avg", "all", "no"], f"Pooling {pooling} not supported"
        self.encoder = encoder
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))

        self.dropout = nn.Dropout(dropout)

        if pooling == "no":
            out_shape = kwargs.get("out_shape", self.encoder.embed_dim)
            assert isinstance(out_shape, int), "out_shape must be an integer"

            self.linear_head = nn.Sequential(
                Rearrange("b n d -> b (n d)"),
                RMSNorm(out_shape),
                self.dropout,
                nn.Linear(out_shape, n_classes),
            )
        else:
            self.linear_head = torch.nn.Sequential(
                RMSNorm(self.encoder.embed_dim),
                self.dropout,
                torch.nn.Linear(self.encoder.embed_dim, n_classes),
            )

        self.pooling = pooling

    def init_weights(self, config_megatron: ConfigInit):
        init_cls(self, config_megatron)
        print("Classifier weights initialized")

    def forward(self, x, pos, return_attn=False):
        if self.pooling == "last_avg":
            x = self.encoder(x, pos, False)
            x = x.mean(dim=1)
            return self.linear_head(x)
        elif self.pooling == "last":
            x = self.encoder(x, pos, False)
        elif self.pooling == "all":  # concatenate all intermediate layers
            x = torch.cat(self.encoder(x, pos, True), dim=1)
        elif self.pooling == "no":
            x = self.encoder(x, pos, False)
            b = x.shape[0]
            query_output = self.cls_query_token.expand(b, -1, -1)
            attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, L)
            context = torch.matmul(attention_weights, x)
            x = torch.cat([context, x], dim=-2)

            return self.linear_head(x)

        b = x.shape[0]
        query_output = self.cls_query_token.expand(b, -1, -1)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, L)
        context = torch.matmul(attention_weights, x).squeeze(1)
        if return_attn:
            return self.linear_head(context), attention_weights

        return self.linear_head(context)

    def forward_attn(self, x, pos):
        # returns prediction, query attention weights, and all intermediate attention weights
        x, attn = self.encoder.forward_attn(x, pos)
        b = x.shape[0]
        query_output = self.cls_query_token.expand(b, -1, -1)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, x).squeeze(1)
        return self.linear_head(context), attention_weights, attn


def get_classifier(args, encoder):
    """
    Get the classifier model for downstream tasks
    The new weights are initialized using the init_cls function.
    """

    kwargs_ = args.task.classifier.kwargs if hasattr(args.task.classifier, "kwargs") else {}
    classifier = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        pooling=args.task.classifier.pooling,
        lp_dropout=args.task.linear_probing.dropout,
        ft_dropout=args.task.fine_tuning.dropout,
        **kwargs_,
    )

    classifier.init_weights(ConfigInit(**args.init))
    return classifier


class ClassifierWrapper(nn.Module):
    """
    A wrapper for the classifier to add a linear layer on top of the encoder.
    This is used for downstream tasks.
    """

    def __init__(self, model, args, h_dim):
        super().__init__()
        self.backbone = model
        # name of classification layer is consistent with the original model
        self.linear_head = nn.Linear(h_dim, args.task.classifier.n_classes)

    def forward(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        return self.linear_head(x)


def wrap_encoder(encoder, args):
    """
    Wrap the model with a classifier for downstream tasks.
    """
    if args.model_type == "reve":
        return get_classifier(args, encoder)
    else:
        h_dim = H_DIM_MAP[args.model_type]
        return ClassifierWrapper(encoder, args, h_dim)
