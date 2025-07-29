from pathlib import Path

import yaml
from classification import evaluate_classification, train_knn_classifier
from data_processing import process_data_batch
from metrics import calculate_clustering_metrics
from model_utils import load_model, setup_model_weights
from visualization import create_pca_visualization


def main():
    """Main function to run the complete inference pipeline."""
    setup_model_weights()

    # Load configuration
    config_file = (
        Path(__file__).parent.parent.parent / "configs" / "inference_config.yaml"
    )
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    model, config = load_model(
        cfg["model_type"],
        cfg["paths"]["config_path"],
        cfg["paths"]["model_path"],
        cfg["device"],
    )

    # Process validation data
    df_results = process_data_batch(
        cfg["paths"]["val_csv"],
        cfg["paths"]["data_path"],
        config,
        model,
        cfg["model_type"],
        cfg["device"],
    )

    # Create visualization and save embeddings
    df_plot = create_pca_visualization(df_results, cfg["output"]["embeddings_results"])

    # Calculate clustering metrics
    clustering_metrics = calculate_clustering_metrics(df_results)

    # Train KNN classifier
    knn_model = train_knn_classifier(
        cfg["paths"]["train_csv"],
        cfg["paths"]["data_path"],
        config,
        model,
        cfg["device"],
    )

    # Evaluate classification
    classification_metrics = evaluate_classification(
        df_plot, knn_model, cfg["output"]["classification_results"]
    )

    return clustering_metrics, classification_metrics


if __name__ == "__main__":
    main()
