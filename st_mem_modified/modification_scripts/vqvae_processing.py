from pathlib import Path

import pandas as pd
from model_utils import load_ecg_data, preprocess_ecg
from tqdm import tqdm
from vqvae_models import extract_embeddings


def process_vqvae_embeddings(model, data_csv, data_path, config, device):
    """Process ECG data to extract VQ-VAE embeddings."""
    df = pd.read_csv(data_csv)
    data_dir = Path(data_path) if data_path else Path(config["dataset"]["ecg_dir"])
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row["RELATIVE_FILE_PATH"]
        file_path = data_dir / file_name

        ecg_data = load_ecg_data(file_path)
        processed_data = preprocess_ecg(ecg_data, config)

        # Ensure 3D: (batch, leads, time)
        if len(processed_data.shape) == 2:
            if processed_data.shape[0] == 2250:  # (time, leads)
                processed_data = processed_data.T.unsqueeze(0)
            else:
                processed_data = processed_data.unsqueeze(0)

        result = extract_embeddings(model.to(device), processed_data.to(device))
        result["file_name"] = file_name
        result["true_label"] = row.get(config["dataset"]["label_col"], None)
        results.append(result)

    return pd.DataFrame(results)
