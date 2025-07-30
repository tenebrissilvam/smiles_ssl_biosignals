# SSL ECG Biosignals Processing

## General task

Self-supervised learning (SSL) has emerged as a powerful
approach for training deep learning models on various data,
including biosinals, such as electrocardiogram (ECG) and
photoplethysmogram (PPG). By leveraging large-scale un-
labeled datasets, SSL enables the pretraining of robust foun-
dation models that can be fine-tuned for downstream tasks
such as arrhythmia detection, cardiovascular risk predic-
tion, and biometric authentication. Recent advances in
SSL demonstrated significant improvements in model gen-
eralizability and performance, even with small annotated
datasets. This paper reviews modifications to SOTA SSL
methods applied to ECG data, emphasizing the benefits
of VQ-VAE-driven representation learning for ECG-specific
tasks and discussing advantages of modality-dependent
over general approaches. Specifically, we propose triplet
loss finetuning, input data representation enhancement and
VQ-VAE as encoder that show improvement in diagnosis
classification task.

Please see SMILES_BIOSIGNALS-1.pdf for schematic explanation of the project.

Full paper text will be availible soon and linked there.

## Setup

### Create Conda Environment
```bash
conda env create -f environment.yml

conda activate ssl-ecg-biosignals
```

### Install Pre-commit Hooks
```bash
pip install pre-commit

pre-commit install
```

### Run Pre-commit on All Files
```bash
pre-commit run --all-files
```

## Project structure

| - ecg_jepa_modified
|
| - st_mem_modified

### st_mem_modified

Spatio-Temporal Masked Electrocardiogram Mod-
eling employs a masked autoencoder to learn spatial-
temporal dependencies in 12-lead ECGs by reconstruct-
ing masked signal segments. Its design explicitly incorpo-
rates the unique spatial (lead-level) and temporal (time-level)
structure of ECG data, distinguishing it from models that pri-
oritize temporal sequences alone.

This is a modification of the ST-MEM method by me, code based on the provided in repository https://github.com/bakqui/ST-MEM

#### run knn classification and clustering evaluation

```bash
cd st_mem_modified/

bash scripts/run_inference_clusters.sh

```

#### run triplet loss model training and inference

```bash

cd st_mem_modified/

bash scripts/run_triplet_loss_train_and_inference.sh

```

#### run vq-vae model training and inference

```bash

cd st_mem_modified/

bash scripts/run_triplet_loss_train_and_inference.sh

```

### ecg_jepa_modified

Electrocardiogram Joint-Embedding Predictive
Architecture learns semantic representations of 12-lead
ECG data via latent space prediction, avoiding noise and
L2 loss limitations by focusing on high-level patterns rather
than raw signal reconstruction. It introduces Cross-Pattern
Attention (CroPA), which restricts attention to patches from
the same ECG lead and time period, mimicking human anal-
ysis to improve efficiency in multi-lead training.

This is a modification of the ECG-JEPA method by Egor Padin, code based on the provided in repository https://github.com/sehunfromdaegu/ECG_JEPA
