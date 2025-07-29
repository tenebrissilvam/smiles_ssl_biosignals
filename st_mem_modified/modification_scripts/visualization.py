import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def create_pca_visualization(df_results, output_path):
    """Create PCA visualization and save results."""
    embeddings = np.stack(
        df_results["logits"].apply(
            lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
        )
    )
    embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
    pca = PCA(n_components=2, random_state=42)
    embedding_2d = pca.fit_transform(embeddings_2d)

    df_results["2d_logits"] = list(embedding_2d)
    df_plot = df_results.dropna()

    x = np.array([point[0] for point in df_plot["2d_logits"]])
    y = np.array([point[1] for point in df_plot["2d_logits"]])

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(df_plot["true_label"])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=numeric_labels, cmap="viridis", alpha=0.6)
    legend1 = plt.legend(
        scatter.legend_elements()[0], label_encoder.classes_, title="Classes"
    )
    plt.gca().add_artist(legend1)
    plt.title("ST-MEM encoder on PTB-XL")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("outputs/pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save results
    df_export = df_results.copy()
    df_export["dim1"] = df_export["2d_logits"].apply(lambda x: x[0])
    df_export["dim2"] = df_export["2d_logits"].apply(lambda x: x[1])
    df_export.drop("2d_logits", axis=1, inplace=True)
    df_export.to_csv(output_path, index=False)

    return df_plot
