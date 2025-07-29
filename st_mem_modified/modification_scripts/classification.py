import numpy as np
import torch
from data_processing import process_data_batch
from metrics import calculate_metrics
from sklearn.neighbors import KNeighborsClassifier


def train_knn_classifier(train_csv, data_path, config, model, device):
    """Train KNN classifier on training data."""
    train_results = process_data_batch(
        train_csv, data_path, config, model, "encoder", device
    )
    cleaned_train = train_results.dropna()

    train_embeds = np.vstack(
        [
            (
                emb.detach().cpu().numpy().flatten()
                if torch.is_tensor(emb)
                else np.array(emb)
            )
            for emb in cleaned_train.logits
        ]
    )

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(train_embeds, cleaned_train.true_label.values)

    return knn


def evaluate_classification(df_plot, knn_model, output_path):
    """Evaluate classification performance."""
    test_embeds = np.vstack(
        [
            (
                emb.detach().cpu().numpy().flatten()
                if torch.is_tensor(emb)
                else np.array(emb)
            )
            for emb in df_plot.logits
        ]
    )

    predicted_classes = knn_model.predict(test_embeds)
    probability_scores = knn_model.predict_proba(test_embeds)

    df_plot["pred_class"] = predicted_classes
    df_plot["probs_CD"] = probability_scores[:, 0]
    df_plot["probs_HYP"] = probability_scores[:, 1]
    df_plot["probs_MI"] = probability_scores[:, 2]
    df_plot["probs_NORM"] = probability_scores[:, 3]
    df_plot["probs_STTC"] = probability_scores[:, 4]

    metrics = calculate_metrics(
        df_plot["true_label"], df_plot["pred_class"], probability_scores
    )

    print(f'F1 score: {metrics["f1_score"]:.3f}')
    print(f'AUROC: {metrics["auroc"]:.3f}')

    df_plot.to_csv(output_path, index=False)

    return metrics
