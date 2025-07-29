from pathlib import Path

import pandas as pd
from model_utils import (
    get_embeddings,
    load_ecg_data,
    preprocess_ecg,
    run_reconstruction_inference,
)
from tqdm import tqdm


def process_data_batch(
    data_csv, data_path, config, model, model_type, device, mask_ratio=0.75
):
    """Process a batch of ECG data and return results."""
    df = pd.read_csv(data_csv)
    data_dir = Path(data_path) if data_path else Path(config["dataset"]["ecg_dir"])
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row[config["dataset"]["inference_file_name"]]
        file_path = data_dir / file_name
        print(f"Processing: {file_path}")

        ecg_data = load_ecg_data(file_path)
        processed_data = preprocess_ecg(ecg_data, config)

        if model_type == "full":
            result = run_reconstruction_inference(
                model, processed_data, mask_ratio, device
            )
        else:
            result = get_embeddings(model, processed_data, device)

        result["file_name"] = file_name
        result["true_label"] = row.get(config["dataset"]["label_col"], None)
        results.append(result)

    return pd.DataFrame(results)
