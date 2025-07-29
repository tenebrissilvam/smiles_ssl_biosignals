import torch
import torch.optim as optim
import yaml
from triplet_models import TripletLoss, TripletModel
from triplet_training import run_final_evaluation, train_triplet_model
from util.dataset import build_dataset, get_dataloader


def load_triplet_config(config_path):
    """Load triplet training configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    # Load triplet training configuration
    triplet_config_path = "/Users/tenebrissilvam/Code/smiles_2025/ssl_ecg_biosignals_processing/configs/triplet_training_config.yaml"
    triplet_config = load_triplet_config(triplet_config_path)

    # Extract configuration values
    device = triplet_config["device"]
    model_config = triplet_config["model"]
    training_config = triplet_config["training"]
    output_config = triplet_config["output"]
    data_config = triplet_config["data"]

    # Load original model configuration
    with open(model_config["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setup datasets and loaders
    train_dataset = build_dataset(config["dataset"], split="train")
    val_dataset = build_dataset(config["dataset"], split="valid")
    train_loader = get_dataloader(
        train_dataset,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )

    # Initialize model
    model = TripletModel(
        model_type=model_config["type"],
        config_path=model_config["config_path"],
        model_path=model_config["encoder_path"],
        device=device,
    ).to(device)

    criterion = TripletLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config["learning_rate"],
    )

    # Train model
    train_triplet_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        training_config["epochs"],
        output_config["output_dir"],
        device,
    )

    # Load best model and run final evaluation
    model = TripletModel(
        model_type=model_config["type"],
        config_path=model_config["config_path"],
        model_path=model_config["encoder_path"],
        device=device,
    ).to(device)

    checkpoint = torch.load(output_config["best_model_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run final evaluation
    metrics = run_final_evaluation(
        model,
        config,
        data_config["data_path"],
        data_config["train_csv"],
        data_config["val_csv"],
        device,
        embeddings_output=output_config["embeddings_output"],
        classification_output=output_config["classification_output"],
    )
    print(f'Final F1 score: {metrics["f1_score"]:.3f}')
    print(f'Final AUROC: {metrics["auroc"]:.3f}')
