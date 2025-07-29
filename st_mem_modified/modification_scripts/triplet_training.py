import os

import torch
from classification import evaluate_classification, train_knn_classifier
from data_processing import process_data_batch
from tqdm import tqdm
from triplet_models import TripletModel
from triplet_utils import evaluate_embeddings, get_triplets
from visualization import create_pca_visualization


def train_triplet_model(
    model, train_loader, val_loader, criterion, optimizer, epochs, output_dir, device
):
    """Train triplet model and save best checkpoints"""
    best_val_f1 = 0.0
    best_val_auroc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for data, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
        ):
            data = data.to(device)
            labels = labels.to(device)

            embeddings = model(data)
            triplets = get_triplets(embeddings, labels)
            if triplets is None:
                continue

            anchors, positives, negatives = triplets
            loss = criterion(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for data, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"
            ):
                data = data.to(device)
                labels = labels.to(device)

                embeddings = model(data)
                triplets = get_triplets(embeddings, labels)
                if triplets is None:
                    continue

                anchors, positives, negatives = triplets
                loss = criterion(anchors, positives, negatives)

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches
        val_f1, val_auroc = evaluate_embeddings(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}"
        )

        # Save best models
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                },
                os.path.join(output_dir, "best_f1_model.pth"),
            )
            print(f"Saved best F1 model: {val_f1:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                },
                os.path.join(output_dir, "best_auroc_model.pth"),
            )
            print(f"Saved best AUROC model: {val_auroc:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )


def evaluate_trained_model(
    model_path, config_path, data_path, data_csv, config, device
):
    """Load trained model and evaluate on test data"""
    model = TripletModel(
        model_type="encoder",
        config_path=config_path,
        model_path="weights/st_mem_vit_base_encoder.pth",
        device=device,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df_results = process_data_batch(
        data_csv, data_path, config, model, "encoder", device
    )
    return df_results


def run_final_evaluation(
    model,
    config,
    data_path,
    train_csv,
    val_csv,
    device,
    embeddings_output=None,
    classification_output=None,
):
    """Run final evaluation with visualization and classification"""
    # Default output paths if not provided
    if embeddings_output is None:
        embeddings_output = "/Users/tenebrissilvam/Code/smiles_2025/ST-MEM/outs/embeddings_results_triplet_loss2.csv"
    if classification_output is None:
        classification_output = "/Users/tenebrissilvam/Code/smiles_2025/ST-MEM/outs/classification_st_mem_cluster_triplet.csv"

    # Process validation data
    val_results = process_data_batch(
        val_csv, data_path, config, model, "encoder", device
    )

    # Create visualization
    df_plot = create_pca_visualization(val_results, embeddings_output)

    # Train KNN on training data and evaluate
    knn_model = train_knn_classifier(train_csv, data_path, config, model, device)
    metrics = evaluate_classification(df_plot, knn_model, classification_output)

    return metrics
