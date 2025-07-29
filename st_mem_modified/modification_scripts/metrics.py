import numpy as np
import torch
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate F1 score and AUROC for classification results."""
    f1 = f1_score(y_true, y_pred, average="weighted")

    auroc = None
    if y_prob is not None:
        classes = np.array(["CD", "HYP", "MI", "NORM", "STTC"])
        y_true_bin = label_binarize(y_true, classes=classes)
        print(y_true_bin)
        auroc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")

    return {"f1_score": f1, "auroc": auroc}


def calculate_clustering_metrics(df_results):
    """Calculate clustering evaluation metrics."""
    label_encoder = LabelEncoder()
    true_clusters = label_encoder.fit_transform(df_results["true_label"])

    embeddings = np.stack(
        df_results["logits"].apply(
            lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
        )
    )
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    nmi = normalized_mutual_info_score(true_clusters, true_clusters)
    silhouette = silhouette_score(embeddings, true_clusters)
    db_index = davies_bouldin_score(embeddings, true_clusters)
    ch_index = calinski_harabasz_score(embeddings, true_clusters)

    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Index: {db_index:.3f}")
    print(f"Calinski-Harabasz Index: {ch_index:.3f}")

    return {
        "nmi": nmi,
        "silhouette": silhouette,
        "db_index": db_index,
        "ch_index": ch_index,
    }
