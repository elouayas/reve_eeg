"""Masked autoencoder training module used for REVE foundation pretraining."""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from models.backbone import get_backbone
from models.encoder import FourierEmb4D, get_reve_model
from utils.initialization import ConfigInit, init_mae


class MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.masking_ratio = config.decoder.masking.ratio
        assert self.masking_ratio > 0 and self.masking_ratio < 1, "masking ratio must be kept between 0 and 1"

        self.encoder = get_reve_model(config, checkpoint=None)  # never load for MAE

        encoder_dim = self.encoder.embed_dim
        pixel_values_per_patch = self.encoder.patch_size

        self.decoder = get_backbone(config.decoder.transformer)

        decoder_dim = self.decoder.dim
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        # if decoder and encoder dim are different remap it below
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.pos_enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.token_avg: bool = config.token_avg

        if self.token_avg:
            self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))
            self.cls_to_pixels = torch.nn.Sequential(
                nn.Linear(encoder_dim, 4 * encoder_dim, bias=False),
                nn.ReLU(),
                nn.Linear(4 * encoder_dim, pixel_values_per_patch),
            )
            self.token_avg_lambda = config.token_avg_lambda

        self.init_weights()

    def init_weights(self):
        config_megatron = ConfigInit(**self.config.init)
        init_mae(self, config_megatron)
        print("MAE weights initialized")

    def forward(self, eeg, pos, b_m=None, b_u=None, return_patches=False):  # noqa: PLR0915
        device = eeg.device
        patches = eeg.unfold(
            dimension=2,
            size=self.encoder.patch_size,
            step=self.encoder.patch_size - self.encoder.overlap_size,
        )
        b, c, h, p = patches.shape
        patches = rearrange(patches, "b c h e -> b (c h) e", c=c, h=h, e=p)
        batch = b
        num_patches = c * h

        if self.training:
            pos = pos + torch.from_numpy(np.random.normal(loc=0, scale=self.encoder.noise_ratio, size=(c, 3))).to(pos)

        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.encoder.ln(self.encoder.fourier4d(pos) + self.encoder.mlp4d(pos))
        tokens = self.encoder.to_patch_embedding(patches) + pos_embed
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        if b_m is None:
            num_masked = int(self.masking_ratio * num_patches)
            if self.training:
                rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
            else:
                torch.manual_seed(42)  # for validation set on same patch if val is done in MAE
                rand_indices = torch.rand(num_patches, device=device).unsqueeze(0).repeat(batch, 1).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        else:
            num_masked = b_m.shape[1]
            masked_indices, unmasked_indices = b_m, b_u
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        # attend with vision transformer

        if self.token_avg and self.training:
            all_outputs = self.encoder.transformer(tokens, True)
            encoded_tokens = all_outputs[-1]
            x = torch.cat(all_outputs, dim=1)
            b = x.shape[0]
            query_output = self.cls_query_token.expand(b, -1, -1)
            key_value_tokens = x
            attention_scores = torch.matmul(query_output, key_value_tokens.transpose(-1, -2)) / (
                self.encoder.embed_dim**0.5
            )
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
        else:
            encoded_tokens = self.encoder.transformer(tokens)
            context = None
        # project encoder to decoder dimensions,
        # if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        # reapply decoder position embedding to unmasked tokens
        decoder_pos_emb = self.pos_enc_to_dec(pos_embed.reshape(b, c * h, -1)).to(pos_embed)
        unmasked_decoder_tokens = decoder_tokens + decoder_pos_emb[batch_range, unmasked_indices]
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + decoder_pos_emb[batch_range, masked_indices]
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder.dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)
        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        # calculate reconstruction loss
        loss = F.l1_loss(pred_pixel_values, masked_patches)

        if self.token_avg and self.training and context is not None:
            repeated_context = (
                repeat(context, "b d -> b n d", n=num_masked) + decoder_pos_emb[batch_range, masked_indices]
            )
            cls_pixels = self.cls_to_pixels(repeated_context)
            loss_cls = F.l1_loss(cls_pixels, masked_patches)
            loss += self.token_avg_lambda * loss_cls

        if return_patches:
            return loss, pred_pixel_values, masked_patches
        else:
            return loss
