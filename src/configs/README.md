# Configuration Guide (Hydra + OmegaConf)

This folder contains all Hydra configuration for pretraining (`src/train.py`) and downstream evaluation/fine-tuning (`src/dt.py`, `src/dt_opt.py`, `src/eval_dt.py`).

## 1) Hydra Basics in This Repository

Hydra composes one final config from:
- A root config file (`config_train.yaml` or `config_dt.yaml`)
- A `defaults` list that pulls files from config groups (`encoder`, `optimizer`, `task`, etc.)
- Command-line overrides (`key=value`)

### Entry points and root configs
- `src/train.py` uses `config_train.yaml`
- `src/dt.py`, `src/dt_opt.py`, `src/eval_dt.py` use `config_dt.yaml`

All these scripts are decorated with `@hydra.main(..., config_path="configs")`, so group names refer to this folder.

### Composition pattern used here
Each root config has a `defaults` section, for example in `config_train.yaml`:
- `encoder: base`
- `decoder: base`
- `optimizer: stable_adamw`
- `scheduler: trapezoid`
- `preprocessing: default`
- `init: mae`

This means Hydra loads the corresponding files:
- `encoder/base.yaml`
- `decoder/base.yaml`
- etc.

For downstream tasks (`config_dt.yaml`), the `defaults` include targeted overrides like:
- `optimizer/stable_adamw@task.linear_probing.optimizer`
- `scheduler/plateau_acc@task.fine_tuning.scheduler`

That syntax injects selected group configs directly into nested fields inside the `task` subtree.

### Working directory behavior
Both root configs set:
- `hydra.job.chdir: true`

So during execution, Python runs inside Hydra’s run directory, not repository root. Outputs (checkpoints, logs, `.hydra/*`) go there unless paths are absolute.

## 2) Folder Structure and What Each Group Controls

### `config_train.yaml`
Pretraining/master config for MAE training.

Main sections:
- `mode`: `train` or `debug`
- `data`: dataset path, workers, subset selection
- `trainer`: epochs, LR, batch size, grad clipping, GPU/node counts
- `checkpointing`: output and state checkpoint settings
- `wandb`: optional experiment tracking
- `hydra`: run directory format

### `config_dt.yaml`
Downstream task training/eval master config.

Main sections:
- `mode`: `train`, `eval_only`, `debug` (current training script behavior is controlled mostly by `training_mode`)
- `pretrained_path`: local file path or Hugging Face shorthand (`hf:brain-bzh/reve-base`)
- `cache_dir`: HF cache location
- `training_mode`: `lp`, `ft`, or `lp+ft`
- `trainer`: dtype/device/grad clip
- `loader`: DataLoader performance knobs
- `task`: task-specific dataset/classifier/stage config loaded from `task/*.yaml`
- `lora`: LoRA switches

### `encoder/`
Backbone architecture variants (`base`, `small`, `tiny`, `large`):
- Transformer width/depth/heads
- EEG tokenization setup (`patch_size`, `patch_overlap`)
- Positional frequencies and noise ratio

### `decoder/`
Decoder architecture for pretraining (`base.yaml`):
- Decoder transformer shape
- Masking behavior (`masking.*`)

### `init/`
Initialization hyperparameters:
- `mae.yaml`: pretraining init settings (includes `num_hidden_layers = encoder.depth + decoder.depth` via resolver)
- `cls_wrapper.yaml`: downstream/classifier wrapper init profile

### `optimizer/`
Optimizer definitions with Hydra instantiate targets:
- `adam`, `adamw`, `sgd`
- `stable_adamw`
- `stable_adamw_distributed`

Each config exposes `_target_` and optimizer kwargs; scripts call `hydra.utils.instantiate(...)`.

### `scheduler/`
LR scheduler configs:
- `trapezoid.yaml` -> custom `utils.optim.CyclicTrapezoidLR`
- `plateau.yaml` and `plateau_acc.yaml` -> `ReduceLROnPlateau` with different modes

### `preprocessing/`
Pretraining-time masking/window config (`default.yaml`):
- Window duration
- Masking ratio/shape/dropout controls

### `task/`
Downstream dataset/task presets (`bciciv2a`, `faced`, `hmc`, `isruc`, `moabb_bciciv2a`, `mumtaz`, `physio`, `speech`, `stress`, `tuab`, `tuev`).

Each task file typically defines:
- `name`, `n_chans`, `duration`
- `classifier`: head size/pooling/dropout
- `data_loader`: dataset `_target_`, path, electrodes, split names, seed, batch size
- `linear_probing`: stage-specific optimizer/scheduler/epochs/patience
- `fine_tuning`: same for FT stage

`_none.yaml` is an empty placeholder used as a safe default when no task is selected yet.

### Utility modules in this folder
- `resolver.py`: custom OmegaConf resolvers registration
- `validate.py`: debug printing / config resolution helpers

## 3) Interpolation and Dynamic Values Used Here

This repository uses OmegaConf interpolations heavily:
- `${env:SCRATCH}`: environment variable expansion
- `${gpu_count:}`: runtime GPU count
- `${cpu_count:}`: runtime CPU count
- `${min:16, ${cpu_count:}}`: nested interpolation
- `${add:${encoder.transformer.depth}, ${decoder.transformer.depth}}`: computed values

These are resolved at runtime after `register_resolvers()` is called.

## 4) Custom OmegaConf Resolvers Implemented

Defined in `src/configs/resolver.py` and registered through `register_resolvers()`.

Important: these are custom resolvers, not built-in Hydra keys.

### `cwd`
- Usage: `${cwd:}`
- Returns: current working directory (`os.getcwd()`)

### `home`, `work`, `scratch`
- Usage: `${home:}`, `${work:}`, `${scratch:}`
- Intended source env vars: `HOME`, `WORK`, `SCRATCH`
- Fallback: `"."` if not set

### `gpu_count`
- Usage: `${gpu_count:}`
- Returns: `torch.cuda.device_count()`

### `cpu_count`
- Usage: `${cpu_count:}`
- Returns: `os.cpu_count()`

### `min`
- Usage: `${min:x,y}`
- Returns: minimum of two values
- Example: `${min:16, ${cpu_count:}}`

### `add`
- Usage: `${add:x,y}`
- Returns: sum of two values
- Example: `${add:${encoder.transformer.depth}, ${decoder.transformer.depth}}`

### `env`
- Usage: `${env:VAR_NAME}`
- Returns: value of environment variable `VAR_NAME`
- Fallback: `"."` if unset

## 5) Common Override Patterns

### Swap architecture variant
```bash
python src/train.py encoder=small
```

### Change optimizer/scheduler groups
```bash
python src/train.py optimizer=adamw scheduler=plateau
```

### Override nested values
```bash
python src/train.py trainer.lr=3e-4 trainer.batch_size=128 data.path=/path/to/preprocessed
```

### Choose downstream task + stage
```bash
torchrun --nproc_per_node=gpu src/dt.py task=bciciv2a training_mode=lp data_root=/path/to/datasets
```

### Override task-stage hyperparameters inline
```bash
torchrun --nproc_per_node=gpu src/dt.py \
  task=bciciv2a \
  training_mode=lp+ft \
  task.linear_probing.n_epochs=50 \
  task.fine_tuning.n_epochs=20
```

## 6) Validation / Debug Behavior

`src/configs/validate.py` provides debug helpers:
- In debug mode, resolved config is written to `.hydra/resolved.yaml`
- Nested config tree is printed via `rich`
- In normal modes, configs are resolved without pretty-printing

This is useful to confirm final composed values after all defaults + overrides + resolvers.

## 7) Notes for Reliable Config Edits

- Keep reusable knobs in config groups (`encoder`, `optimizer`, etc.) and task-specific knobs in `task/*.yaml`.
- Prefer command-line overrides for temporary experiments; commit YAML changes only when they should become defaults.
- When adding new dynamic interpolation, register a resolver in `resolver.py` and call `register_resolvers()` before config resolution in the script.
- For Optuna (`src/dt_opt.py`), task configs may include strings like `optuna:float:init:low:high:log`; these are parsed by `utils/model_utils.py` before trials run.
