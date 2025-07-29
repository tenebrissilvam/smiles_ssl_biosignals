# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
_PTB_XL_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the WFDB ECG database directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the preprocessed signals will be saved.",
    )
    parser.add_argument(
        "--index_path", type=str, default="./index.csv", help="Path to the index file."
    )
    parser.add_argument(
        "--database_type",
        type=str,
        choices=["standard", "ptb-xl"],
        help="Explicitly specify the database type. If not provided, will try to auto-detect.",
    )
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories.
    Args:
        root_dir (str): The directory to search for .hea files.
    Returns:
        records (set): A set of record names.
                       (e.g., ['database/1/ecg001', 'database/1/ecg001', ..., 'database/9/ecg991'])
    """
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == ".hea":
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records


def find_ptb_xl_records(root_dir):
    """Find all the .hea files in the PTB-XL database structure.
    PTB-XL has a different structure with one less folder depth.

    Args:
        root_dir (str): The directory to search for .hea files (PTB-XL records500 folder).
    Returns:
        records (set): A set of record names.
                       (e.g., ['00000/00001_hr', '00000/00002_hr', ..., '21000/21837_hr'])
    """
    records = set()
    # PTB-XL has a structure with numbered folders (00000, 01000, etc.) containing ECG files
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hea"):
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records


def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window."""
    if crop_length > x.shape[1]:
        raise ValueError(
            f"crop_length must be smaller than the length of x ({x.shape[1]})."
        )
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i : i + crop_length] for i in start_idx]


def process_standard_database(args):
    """Process standard ECG database with person-level folder structure."""
    # Identify the header files
    record_rel_paths = find_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} records.")

    # Prepare an index dataframe
    index_df = pd.DataFrame(
        columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE"]
    )

    # Save all the cropped signals
    num_saved = 0
    num_skipped = 0
    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = os.path.join(args.output_dir, record_rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_name = record_rel_dir.split("/")[0]

        try:
            signal, record_info = wfdb.rdsamp(
                os.path.join(args.input_dir, record_rel_path)
            )
        except (ValueError, Exception) as e:
            # Skip records with malformed headers or date format issues
            num_skipped += 1
            continue

        try:
            lead_idx = np.array(
                [record_info["sig_name"].index(lead_name) for lead_name in _LEAD_NAMES]
            )
        except ValueError:
            # Skip records that don't have all required leads
            num_skipped += 1
            continue

        signal = signal[:, lead_idx]
        fs = record_info["fs"]
        signal_length = record_info["sig_len"]
        if (
            signal_length < 10 * fs
        ):  # Exclude the ECGs with lengths of less than 10 seconds
            continue
        cropped_signals = moving_window_crop(
            signal.T, crop_length=10 * fs, crop_stride=10 * fs
        )
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                continue
            pd.to_pickle(
                cropped_signal.astype(np.float32),
                os.path.join(save_dir, f"{record_name}_{idx}.pkl"),
            )
            index_df.loc[num_saved] = [
                f"{record_rel_path}_{idx}.pkl",
                f"{record_name}_{idx}.pkl",
                fs,
                source_name,
            ]
            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")
    print(f"Skipped {num_skipped} records due to errors or missing leads.")
    return index_df


def process_ptb_xl_database(args):
    """Process PTB-XL database which has a different folder structure."""
    # Identify the header files in PTB-XL structure
    record_rel_paths = find_ptb_xl_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} PTB-XL records.")

    # Prepare an index dataframe
    index_df = pd.DataFrame(
        columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE"]
    )

    # Save all the cropped signals
    num_saved = 0
    num_skipped = 0
    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = os.path.join(args.output_dir, record_rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_name = "ptb-xl"  # Set source name for PTB-XL database

        try:
            signal, record_info = wfdb.rdsamp(
                os.path.join(args.input_dir, record_rel_path)
            )
        except (ValueError, Exception) as e:
            # Skip records with malformed headers or date format issues
            num_skipped += 1
            continue

        try:
            # Use PTB-XL specific lead names (uppercase AVR, AVL, AVF instead of aVR, aVL, aVF)
            lead_idx = np.array(
                [
                    record_info["sig_name"].index(lead_name)
                    for lead_name in _PTB_XL_LEAD_NAMES
                ]
            )
        except ValueError:
            # Skip records that don't have all required leads
            num_skipped += 1
            continue

        signal = signal[:, lead_idx]
        fs = record_info["fs"]
        signal_length = record_info["sig_len"]
        if (
            signal_length < 10 * fs
        ):  # Exclude the ECGs with lengths of less than 10 seconds
            continue
        cropped_signals = moving_window_crop(
            signal.T, crop_length=10 * fs, crop_stride=10 * fs
        )
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                continue
            pd.to_pickle(
                cropped_signal.astype(np.float32),
                os.path.join(save_dir, f"{record_name}_{idx}.pkl"),
            )
            index_df.loc[num_saved] = [
                f"{record_rel_path}_{idx}.pkl",
                f"{record_name}_{idx}.pkl",
                fs,
                source_name,
            ]
            num_saved += 1

    print(f"Saved {num_saved} PTB-XL cropped signals.")
    print(f"Skipped {num_skipped} PTB-XL records due to errors or missing leads.")
    return index_df


def run(args):
    # Determine the database type
    is_ptb_xl = False

    # Use explicitly provided database_type if available
    if hasattr(args, "database_type") and args.database_type:
        is_ptb_xl = args.database_type == "ptb-xl"
        print(f"Using specified database type: {args.database_type}")
    # Otherwise try to auto-detect
    elif os.path.basename(args.input_dir) == "records500" or re.search(
        r"ptb-xl", args.input_dir, re.IGNORECASE
    ):
        is_ptb_xl = True
        print("Auto-detected PTB-XL database structure.")

    # Process the appropriate database type
    if is_ptb_xl:
        index_df = process_ptb_xl_database(args)
    else:
        index_df = process_standard_database(args)

    # Save the index file
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    index_df.to_csv(args.index_path, index=False)


if __name__ == "__main__":
    run(get_parser())
