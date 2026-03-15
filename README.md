# REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects

[![arXiv](https://img.shields.io/badge/arXiv-2510.21585-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.21585)
[![Website](https://img.shields.io/badge/Website-brain--bzh.github.io-0A66C2?logo=github&logoColor=white)](https://brain-bzh.github.io/reve/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Tutorial-orange?logo=google-colab&logoColor=orange)](https://colab.research.google.com/drive/13wMytamrucmU9c2OIc-TcoEIj6xSqPGo?usp=sharing)
[![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-REVE%20Models-FFD21E?logo=huggingface&logoColor=yellow)](https://huggingface.co/collections/brain-bzh/reve)


Codebase for REVE (EEG foundation model):
- Foundation pretraining (MAE-style) in `src/train.py`
- Downstream linear probing / fine-tuning in `src/dt.py`
- Optuna search for downstream hyperparameters in `src/dt_opt.py`
- Evaluation utility in `src/eval_dt.py`

## Repository layout

- `src/`: training, downstream, models, and utilities
- `src/configs/`: Hydra configuration tree for all runs
- `preprocessing/`: dataset conversion/preparation scripts
- `hf/`: Hugging Face export artifacts and wrappers
- `.scripts/`: helper launch scripts

Detailed docs:
- [Configs guide](src/configs/README.md)
- [Downstream tasks guide](src/downstream_tasks/README.md)
- [Preprocessing guide](preprocessing/README.md)
- [Hugging Face export notes](hf/README.md)

## Environment setup

Python requirement comes from `pyproject.toml` (`>=3.11`).

Using `uv`:

```bash
uv sync
```

Using `pip` (alternative):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run scripts directly with `python`, `accelerate`, or `torchrun`.

## How runs are configured

All training/eval scripts use Hydra configs from `src/configs`.

- Pretraining root config: `src/configs/config_train.yaml`
- Downstream root config: `src/configs/config_dt.yaml`

Read the full config documentation here:
- [src/configs/README.md](src/configs/README.md)

## Running scripts

### 1) Pretraining

Main entrypoint:

```bash
accelerate launch src/train.py --config-name config_train.yaml
```

Common overrides:

```bash
accelerate launch src/train.py --config-name config_train.yaml \
  data.path=/path/to/preprocessed_foundation_data \
  checkpointing.path=/path/to/checkpoints \
  trainer.batch_size=256 \
  trainer.lr=3e-4
```

Switch encoder variant:

```bash
accelerate launch src/train.py --config-name config_train.yaml encoder=small
```

Helper wrapper:

```bash
bash .scripts/train.sh
```

### 2) Downstream training (LP / FT)

Main entrypoint:

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=hf:brain-bzh/reve-base \
  training_mode=lp
```

Run both stages (linear probing then fine-tuning):

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=/path/to/encoder.pth \
  training_mode=lp+ft
```

Adjust stage hyperparameters inline:

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  training_mode=lp \
  task.linear_probing.n_epochs=50 \
  task.linear_probing.optimizer.lr=1e-3
```

Helper wrapper for all tasks:

```bash
bash .scripts/dt.sh /path/to/downstream_data
```

### 3) Downstream Optuna search

```bash
torchrun --nproc_per_node=gpu src/dt_opt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  pretrained_path=hf:brain-bzh/reve-base \
  n_runs=50
```

`dt_opt.py` reads `optuna:*` parameter declarations from config values (for example in `task/bciciv2a.yaml`).

### 4) Evaluate a saved downstream checkpoint

By default `src/eval_dt.py` loads `model_best.pth` from current run directory.

```bash
python src/eval_dt.py \
  --config-name config_dt.yaml \
  task=bciciv2a \
  data_root=/path/to/downstream_data \
  checkpoint_path=/path/to/model_best.pth
```

### 5) Preprocessing scripts

Preprocessing scripts live in `preprocessing/` and are dataset-specific.

Examples:

```bash
python preprocessing/preprocessing_bciciv2a.py --raw /path/raw --processed /path/processed
python preprocessing/preprocessing_physio.py --root /path/raw --processed /path/processed --type reve
python preprocessing/ISRUC/prepare_ISRUC.py --raw /path/raw --processed /path/out/{}
```

See full notes here:
- [preprocessing/README.md](preprocessing/README.md)

## Dataset

Part of the dataset (for which the license allows redistribution) is available on Hugging Face:

🔗 [brain-bzh/reve-dataset](https://huggingface.co/datasets/brain-bzh/reve-dataset)

More documentation on how to use the dataset will be provided soon.

## Notes

- Hydra sets run directories (see `hydra.run.dir` in config files), and scripts use `hydra.job.chdir: true`.
- Set environment variables expected by config interpolations (for example `SCRATCH`) if you rely on `${env:SCRATCH}` defaults.
- Downstream tasks and dataset metadata are defined in `src/configs/task/*.yaml`.

## Contributors

Special thanks to all the people who contributed to the codebase.
- Timotée Ly-Manson
- Alix Lamouroux

## Citation

If you use this repository or pretrained REVE models, cite:

```bibtex
@article{elouahidi2025reve,
    title   = {{REVE}: A Foundation Model for {EEG}: Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects},
    author  = {El Ouahidi, Yassine and Lys, Jonathan and Thölke, Philipp and Farrugia, Nicolas and Pasdeloup, Bastien and Gripon, Vincent and Jerbi, Karim and Lioi, Giulia},
    journal = {Advances in Neural Information Processing Systems},
    year    = {2025},
    url     = {https://brain-bzh.github.io/reve/}
}
```

Machine-readable citation metadata is available in `CITATION.cff`.
