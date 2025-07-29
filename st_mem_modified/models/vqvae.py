# Copyright 2024
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# VQ-VAE: https://arxiv.org/abs/1711.00937
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from models.encoder.vit import TransformerBlock


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings with better initialization
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.1)

        # EMA for codebook updates
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.zeros(num_embeddings, embedding_dim))
        self._decay = 0.99
        self._epsilon = 1e-5

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = (
            inputs.permute(0, 2, 3, 1).contiguous() if inputs.dim() == 4 else inputs
        )
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings).view(input_shape)

        # EMA update during training
        if self.training:
            self._ema_cluster_size = self._decay * self._ema_cluster_size + (
                1 - self._decay
            ) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw
            self.embeddings.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Convert quantized from BHWC -> BCHW
        if quantized.dim() == 4:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, encoding_indices.unsqueeze(1)


class ECG_Encoder(nn.Module):
    """
    Encoder for ECG data
    """

    def __init__(
        self,
        seq_len,
        patch_size,
        num_leads,
        width=768,
        depth=12,
        mlp_dim=3072,
        heads=12,
        dim_head=64,
        qkv_bias=True,
        latent_dim=256,
    ):
        super().__init__()

        assert (
            seq_len % patch_size == 0
        ), "The sequence length must be divisible by the patch size."

        # Embedding layers
        num_patches = seq_len // patch_size
        patch_dim = num_leads * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)", p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, width),
            nn.LayerNorm(width),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, width))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=width,
                    output_dim=width,
                    hidden_dim=mlp_dim,
                    heads=heads,
                    dim_head=dim_head,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(width)

        # Projection to latent space
        self.projection = nn.Linear(width, latent_dim)

    def forward(self, x):
        # x shape: [batch_size, num_leads, seq_len]
        x = self.to_patch_embedding(x)
        # x shape: [batch_size, num_patches, width]

        x = x + self.pos_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Project to latent space
        z = self.projection(x)

        return z


class ECG_Decoder(nn.Module):
    """
    Decoder for ECG data
    """

    def __init__(
        self,
        seq_len,
        patch_size,
        num_leads,
        width=768,
        depth=4,
        mlp_dim=3072,
        heads=12,
        dim_head=64,
        qkv_bias=True,
        latent_dim=256,
    ):
        super().__init__()

        assert (
            seq_len % patch_size == 0
        ), "The sequence length must be divisible by the patch size."

        num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.num_leads = num_leads
        self.seq_len = seq_len

        # Projection from latent space
        self.projection = nn.Linear(latent_dim, width)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, width))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=width,
                    output_dim=width,
                    hidden_dim=mlp_dim,
                    heads=heads,
                    dim_head=dim_head,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(width)

        # Output layer
        self.output = nn.Linear(width, patch_size * num_leads)

    def forward(self, z):
        # z shape: [batch_size, num_patches, latent_dim]
        x = self.projection(z)
        # x shape: [batch_size, num_patches, width]

        x = x + self.pos_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Output layer
        x = self.output(x)

        # Reshape to ECG format
        x = rearrange(x, "b n (p c) -> b c (n p)", p=self.patch_size, c=self.num_leads)

        return x


class VQVAE(nn.Module):
    """
    VQ-VAE model for ECG data
    """

    def __init__(
        self,
        seq_len=2250,
        patch_size=75,
        num_leads=12,
        encoder_width=768,
        encoder_depth=12,
        decoder_width=512,
        decoder_depth=4,
        mlp_ratio=4,
        num_heads=12,
        dim_head=64,
        qkv_bias=True,
        latent_dim=256,
        num_embeddings=512,
        commitment_cost=0.25,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_leads = num_leads

        # Encoder
        self.encoder = ECG_Encoder(
            seq_len=seq_len,
            patch_size=patch_size,
            num_leads=num_leads,
            width=encoder_width,
            depth=encoder_depth,
            mlp_dim=encoder_width * mlp_ratio,
            heads=num_heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            latent_dim=latent_dim,
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder
        self.decoder = ECG_Decoder(
            seq_len=seq_len,
            patch_size=patch_size,
            num_leads=num_leads,
            width=decoder_width,
            depth=decoder_depth,
            mlp_dim=decoder_width * mlp_ratio,
            heads=num_heads // 2,  # Smaller decoder
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            latent_dim=latent_dim,
        )

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        loss = recon_loss + vq_loss

        return x_recon, loss, vq_loss, indices


def vqvae_small(**kwargs):
    model_args = dict(
        encoder_width=384,
        encoder_depth=12,
        decoder_width=256,
        decoder_depth=4,
        num_heads=6,
        mlp_ratio=4,
        latent_dim=192,
        num_embeddings=512,
        **kwargs,
    )
    return VQVAE(**model_args)


def vqvae_base(**kwargs):
    model_args = dict(
        encoder_width=768,
        encoder_depth=12,
        decoder_width=512,
        decoder_depth=4,
        num_heads=12,
        mlp_ratio=4,
        latent_dim=256,
        num_embeddings=1024,
        **kwargs,
    )
    return VQVAE(**model_args)
