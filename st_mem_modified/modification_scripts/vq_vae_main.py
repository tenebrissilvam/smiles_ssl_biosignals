import torch
import torch.backends.cudnn as cudnn
import yaml
from classification import evaluate_classification, train_knn_classifier
from metrics import calculate_clustering_metrics
from visualization import create_pca_visualization

# Import functions from other modules
from vqvae_models import load_vqvae_encoder, setup_vqvae_model
from vqvae_processing import process_vqvae_embeddings
from vqvae_training import train_vqvae_model


def main():
    # Load VQ-VAE specific config
    vqvae_config_path = "/Users/tenebrissilvam/Code/smiles_2025/ssl_ecg_biosignals_processing/configs/vqvae_config.yaml"
    with open(vqvae_config_path, "r") as f:
        vqvae_config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model config
    with open(vqvae_config["paths"]["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["output_dir"] = vqvae_config["paths"]["output_dir"]
    config["exp_name"] = vqvae_config["experiment"]["exp_name"]

    device = vqvae_config["experiment"]["device"]
    torch.manual_seed(config["seed"])
    cudnn.benchmark = True

    # Setup and train model
    model = setup_vqvae_model(config, device)
    train_vqvae_model(model, config, device)

    # Load trained model for evaluation
    model = load_vqvae_encoder(vqvae_config["paths"]["checkpoint_path"])

    # Process embeddings
    df_results = process_vqvae_embeddings(
        model,
        vqvae_config["paths"]["val_csv"],
        vqvae_config["paths"]["data_path"],
        config,
        device,
    )

    # Visualization and evaluation
    df_plot = create_pca_visualization(
        df_results, vqvae_config["outputs"]["embeddings_csv"]
    )
    calculate_clustering_metrics(df_results)

    # Classification evaluation
    knn_model = train_knn_classifier(
        vqvae_config["paths"]["train_csv"],
        vqvae_config["paths"]["data_path"],
        config,
        model,
        device,
    )
    metrics = evaluate_classification(
        df_plot, knn_model, vqvae_config["outputs"]["classification_csv"]
    )


if __name__ == "__main__":
    main()
