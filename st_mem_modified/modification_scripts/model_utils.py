import pickle

import models as st_models
import models.encoder as encoder
import torch
import torch.nn.functional as F
import yaml
from torchvision import models
from util.transforms import Compose, ToTensor, get_transforms_from_config


def setup_model_weights():
    """Setup missing model weight classes."""
    if not hasattr(models, "VGG16_Weights"):

        class VGG16_Weights:
            IMAGENET1K_V1 = None

        models.VGG16_Weights = VGG16_Weights

    weight_classes = [
        "ResNet18_Weights",
        "ResNet34_Weights",
        "ResNet50_Weights",
        "ResNet101_Weights",
        "ResNet152_Weights",
        "VGG19_Weights",
    ]

    for weight_class in weight_classes:
        if not hasattr(models, weight_class):
            WeightClass = type(weight_class, (), {"IMAGENET1K_V1": None})
            setattr(models, weight_class, WeightClass)


def load_model(model_type, config_path, model_path, device):
    """Load model from checkpoint."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if model_type == "full":
        model_name = config.get("model_name", "st_mem_vit_base_dec256d4b")
        if hasattr(st_models, model_name):
            model = getattr(st_models, model_name)(**config["model"])
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    elif model_type == "encoder":
        model_name = config.get("model_name", "st_mem_vit_base")
        if model_name in encoder.__dict__:
            model = encoder.__dict__[model_name](**config["model"])
        else:
            raise ValueError(f"Unsupported encoder model name: {model_name}")
    else:
        raise ValueError("model_type must be 'full' or 'encoder'")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded with message: {msg}")

    model.to(device)
    model.eval()

    return model, config


def load_ecg_data(file_path):
    """Load ECG data from pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def preprocess_ecg(data, config):
    """Apply preprocessing transforms to ECG data."""
    transforms_config = config["dataset"].get("eval_transforms", [])
    transforms = get_transforms_from_config(transforms_config)
    transforms = Compose(transforms + [ToTensor()])

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    data = transforms(data)
    return data


def run_reconstruction_inference(model, data, mask_ratio=0.75, device="cpu"):
    """Run reconstruction inference using full ST-MEM model."""
    model.eval()
    with torch.no_grad():
        if data.dim() == 2:
            data = data.unsqueeze(0)

        data = data.to(device)
        result = model(data, mask_ratio=mask_ratio)

        pred = result["pred"]
        mask = result["mask"]
        loss = result["loss"] if "loss" in result else None

        reconstructed = model.unpatchify(pred)

        return {
            "original": data.cpu(),
            "reconstructed": reconstructed.cpu(),
            "mask": mask.cpu(),
            "loss": loss.item() if loss is not None else None,
        }


def get_embeddings(model, data, device="cpu"):
    """Run inference using encoder model to get embeddings."""
    model.eval()
    with torch.no_grad():
        if data.dim() == 4:
            batch_size, n_crops = data.shape[:2]
            logits_list = []

            for i in range(n_crops):
                crop_data = data[:, i].to(device)
                logits = model(crop_data)
                logits_list.append(logits)

            logits_tensor = torch.stack(logits_list, dim=1)
            logits = logits_tensor.mean(dim=1)
        else:
            if data.dim() == 2:
                data = data.unsqueeze(0)

            data = data.to(device)
            logits = model(data)

        return {"logits": logits.cpu()}
