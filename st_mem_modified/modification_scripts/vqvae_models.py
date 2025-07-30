import models.encoder as encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_vqvae import EncoderVQVAE


def extract_embeddings(model, processed_data):
    """Extract quantized embeddings from ECG data."""
    with torch.no_grad():
        z_q, vq_loss, indices = model.encode(processed_data)
    return {"embeddings": z_q.cpu().numpy(), "indices": indices.cpu().numpy()}


def reconstruct_ecg(model, ecg_data):
    """Reconstruct ECG from input data."""
    with torch.no_grad():
        x_recon, total_loss, vq_loss, indices = model(ecg_data)
    return x_recon


class VectorQuantizerFixed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize with uniform distribution
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embedding_dim) - 0.5)

    def forward(self, inputs):
        flat_input = inputs.view(-1, self.embedding_dim)

        # L2 distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings[encoding_indices].view(inputs.shape)

        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class EncoderVQVAEFixed(nn.Module):
    def __init__(
        self,
        pretrained_encoder,
        latent_dim=128,
        num_embeddings=512,
        commitment_cost=0.1,
    ):
        super().__init__()

        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_dim = self.encoder.width

        # No activation on projection
        self.to_latent = nn.Linear(encoder_dim, latent_dim)
        self.vq = VectorQuantizerFixed(num_embeddings, latent_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 2250 * 12),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.forward_encoding(x)

        z = self.to_latent(features)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q).view(-1, 12, 2250)

        recon_loss = F.mse_loss(x_recon, x)

        return x_recon, recon_loss + vq_loss, vq_loss, indices


def setup_vqvae_model(config, device):
    """Setup VQ-VAE model with pretrained encoder."""
    pretrained_path = config["model"]["pretrained_encoder_path"]
    checkpoint = torch.load(pretrained_path, map_location=device)

    pretrained_encoder = encoder.st_mem_vit_base(
        num_leads=config["model"]["num_leads"], seq_len=config["model"]["seq_len"]
    )
    pretrained_encoder.load_state_dict(checkpoint["model"], strict=False)

    model = EncoderVQVAE(
        pretrained_encoder=pretrained_encoder,
        latent_dim=config["model"]["latent_dim"],
        num_embeddings=config["model"]["num_embeddings"],
        commitment_cost=config["model"]["commitment_cost"],
        seq_len=config["model"]["seq_len"],
        num_leads=config["model"]["num_leads"],
    ).to(device)

    return model


def load_vqvae_encoder(checkpoint_path, device="auto"):
    """Load trained VQ-VAE model and return encoder part."""

    device = "mps"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Create base encoder (frozen part)
    base_encoder = encoder.st_mem_vit_base(
        num_leads=model_config.get("num_leads", 12),
        seq_len=model_config.get("seq_len", 2500),
    )

    # Create full VQ-VAE model
    vqvae_model = EncoderVQVAE(
        pretrained_encoder=base_encoder,
        latent_dim=model_config.get("latent_dim", 256),
        num_embeddings=model_config.get("num_embeddings", 512),
        commitment_cost=model_config.get("commitment_cost", 0.25),
        seq_len=model_config.get("seq_len", 2500),
        num_leads=model_config.get("num_leads", 12),
    )

    # Load trained weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    vqvae_model.load_state_dict(state_dict, strict=False)
    vqvae_model.to(device)
    vqvae_model.eval()

    return vqvae_model
