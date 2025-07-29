import numpy as np
import torch
from metrics import calculate_metrics
from sklearn.neighbors import KNeighborsClassifier


def get_triplets(embeddings, labels):
    """Generate triplets (anchor, positive, negative) from embeddings and labels"""
    batch_size = embeddings.shape[0]
    triplets = []

    for i in range(batch_size):
        anchor = embeddings[i].unsqueeze(0)
        anchor_label = labels[i].item()

        # Find positive samples (same label)
        pos_indices = torch.where(labels == anchor_label)[0]
        pos_indices = pos_indices[pos_indices != i]  # Exclude the anchor itself

        # Find negative samples (different label)
        neg_indices = torch.where(labels != anchor_label)[0]

        # Skip if no positives or negatives
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # Select a random positive and negative
        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]

        positive = embeddings[pos_idx].unsqueeze(0)
        negative = embeddings[neg_idx].unsqueeze(0)

        triplets.append((anchor, positive, negative))

    if not triplets:
        return None

    # Stack all triplets
    anchors = torch.cat([t[0] for t in triplets], dim=0)
    positives = torch.cat([t[1] for t in triplets], dim=0)
    negatives = torch.cat([t[2] for t in triplets], dim=0)

    return anchors, positives, negatives


def evaluate_embeddings(model, val_loader, device, num_classes=5):
    """Extract embeddings and evaluate with KNN classifier"""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            embeddings = model(data)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    split_idx = int(0.8 * len(embeddings))
    train_emb, test_emb = embeddings[:split_idx], embeddings[split_idx:]
    train_lbl, test_lbl = labels[:split_idx], labels[split_idx:]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_emb, train_lbl)
    pred_proba = knn.predict_proba(test_emb)
    pred_labels = knn.predict(test_emb)

    metrics = calculate_metrics(test_lbl, pred_labels, pred_proba)
    return metrics["f1_score"], metrics["auroc"]
