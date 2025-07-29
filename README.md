# SSL ECG Biosignals Processing

## Project structure

| - ecg_jepa_modified
|
| - st_mem_modified

### st_mem_modified

modification of the ST-MEM method by me, code based on the provided in repository https://github.com/bakqui/ST-MEM


### ecg_jepa_modified

modification of the ECG-JEPA method by Egor Padin, code based on the provided in repository https://github.com/sehunfromdaegu/ECG_JEPA


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

## Development

### Code Formatting
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

### Running Tests
```bash
pytest tests/
```
