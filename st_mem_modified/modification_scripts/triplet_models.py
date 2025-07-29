import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import load_model


class TripletModel(nn.Module):
    def __init__(self, model_type, config_path, model_path, device):
        super(TripletModel, self).__init__()
        self.encoder, self.config = load_model(
            model_type, config_path, model_path, device
        )

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add two additional layers
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        with torch.set_grad_enabled(
            not self.encoder.parameters().__next__().requires_grad
        ):
            features = self.encoder.forward_encoding(x)

        x = self.fc1(features)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.normalize(x, p=2, dim=1)

        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
