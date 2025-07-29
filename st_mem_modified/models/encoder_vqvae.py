#!/usr/bin/env python3
"""
VQ-VAE wrapper for pretrained ST-MEM encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vqvae import VectorQuantizer


class EncoderVQVAE(nn.Module):
    """VQ-VAE using pretrained ST-MEM encoder."""

    def __init__(
        self,
        pretrained_encoder,
        latent_dim=256,
        num_embeddings=512,
        commitment_cost=0.25,
        seq_len=2250,
        num_leads=12,
    ):
        super().__init__()

        # Freeze pretrained encoder
        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get encoder output dimension
        encoder_dim = self.encoder.width

        # Projection to latent space
        self.to_latent = nn.Linear(encoder_dim, latent_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, seq_len * num_leads),
        )

        self.seq_len = seq_len
        self.num_leads = num_leads

    def encode(self, x):
        # Get features from pretrained encoder
        with torch.no_grad():
            features = self.encoder.forward_encoding(x)

        # Project to latent space
        z = self.to_latent(features)
        z_q, vq_loss, indices = self.vq(z.unsqueeze(-1).unsqueeze(-1))

        return z_q.squeeze(-1).squeeze(-1), vq_loss, indices.squeeze(-1)

    def decode(self, z_q):
        return self.decoder(z_q).view(-1, self.num_leads, self.seq_len)

    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        return x_recon, recon_loss + vq_loss, vq_loss, indices
