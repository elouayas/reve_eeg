# Downstream Tasks Guide

This folder contains dataset loaders and utilities used by downstream training/evaluation scripts:
- `src/dt.py`
- `src/dt_opt.py`
- `src/eval_dt.py`

## What lives here

- `dataloaders.py`: shared loader factory + `LMDBDataset` + `NeuroLMDataset`
- `dataloader_moabb.py`: MOABB-backed dataset wrapper (`MOABBDataset`)
- `dataloader_tuh.py`: TUAB/TUEV dataset wrappers
- `dataloader_isruc.py`: ISRUC loader
- `position_utils.py`: electrode position loading from file or HF position bank
- `train_core.py`, `eval_core.py`, `utils.py`: helper/legacy training utilities

## Data flow in downstream runs

1. `src/dt.py` reads a task config from `src/configs/task/*.yaml`.
2. `get_data_loaders(...)` in `dataloaders.py` instantiates the dataset via Hydra (`config.dataset._target_`).
3. The dataset is instantiated three times with `mode` set from `splits` (default: `train`, `val`, `test`).
4. Each dataset must provide:
- `__getitem__` returning sample + label (and optionally `pos`)
- `collate(batch)` returning a dict with `sample`, `label`, `pos`
5. `src/dt.py` consumes each batch as:
- `data = batch["sample"]`
- `target = batch["label"]`
- `pos = batch["pos"]`

## Batch contract (important)

Your dataset integration must produce batches with this structure:

- `sample`: `torch.Tensor` of shape `[B, C, T]` (or compatible tensor expected by model)
- `label`: integer class tensor (typically `[B]`)
- `pos`: `torch.Tensor` of shape `[B, C, 3]` with electrode positions

Notes:
- Current training code does `target.long()` and uses `CrossEntropyLoss`, so labels must be class indices.
- `pos` is required by `ReveClassifier` forward path.
- Per-task config values should match dataset output (`task.n_chans`, `task.duration`, `task.classifier.n_classes`).

## Add a new dataset via Hydra config (no new Python class)

Use this path when one of the existing dataset classes already fits your format.

## 1) Pick an existing loader class

Available reusable loaders:
- `downstream_tasks.dataloaders.LMDBDataset`
- `downstream_tasks.dataloaders.NeuroLMDataset`
- `downstream_tasks.dataloader_moabb.MOABBDataset`
- `downstream_tasks.dataloader_tuh.TUAB`
- `downstream_tasks.dataloader_tuh.TUEV`
- `downstream_tasks.dataloader_isruc.ISRUCDataset`

## 2) Create a new task config

Add `src/configs/task/<your_task>.yaml` with at minimum:
- `name`
- `n_chans`
- `duration`
- `classifier` (`n_classes`, `pooling`, optional `dropout`)
- `data_loader` (`dataset`, `batch_size`, `seed`, `splits`)
- `linear_probing` block
- `fine_tuning` block

Minimal pattern:

```yaml
name: MyDataset
n_chans: 32
duration: 5

classifier:
  n_classes: 4
  pooling: "no"

data_loader:
  dataset:
    _target_: downstream_tasks.dataloaders.LMDBDataset
    path: ${data_root}/my_dataset/processed
    electrodes: ["Fz", "Cz", "Pz", ...]
    scale_factor: 100.0
  batch_size: 64
  seed: ${seed}
  splits: ["train", "val", "test"]

linear_probing:
  n_epochs: 20
  patience: 5
  warmup_epochs: 3
  mixup: True
  optimizer:
    lr: 5e-3

fine_tuning:
  n_epochs: 100
  patience: 10
  warmup_epochs: 5
  mixup: True
  optimizer:
    lr: 1e-4

clip: 100
```

Run with:

```bash
torchrun --nproc_per_node=gpu src/dt.py \
  --config-name config_dt.yaml \
  task=<your_task> \
  data_root=/path/to/data \
  pretrained_path=hf:brain-bzh/reve-base \
  training_mode=lp
```

## Add a new MOABB dataset via config (recommended when possible)

If your dataset exists in MOABB, you usually do not need to write a new loader.
Use `downstream_tasks.dataloader_moabb.MOABBDataset` and pass the MOABB dataset class in config.

Template:

```yaml
data_loader:
  dataset:
    _target_: downstream_tasks.dataloader_moabb.MOABBDataset
    _recursive_: False
    dataset_kwargs:
      _target_: moabb.datasets.BNCI2014_001

    electrodes: ["Fz", "FC3", "FC1", ...]
    scale_factor: 1000.0

    paradigm_kwargs:
      tmin: 2.0
      tmax: 5.996
      resample: 200

    label_map:
      left_hand: 0
      right_hand: 1
      feet: 2
      tongue: 3

    slices:
      train: [0, 5]
      val: [5, 7]
      test: [7, 9]

  batch_size: 64
  seed: ${seed}
  splits: ["train", "val", "test"]
```

Why `_recursive_: False` here:
- `MOABBDataset` expects `dataset_kwargs` as a Hydra config object and instantiates it internally (`hydra.utils.instantiate(dataset_kwargs)`).

MOABB-specific checks:
- Ensure `label_map` exactly matches normalized labels returned by MOABB (lowercased + spaces replaced with `_`).
- Ensure `electrodes` order matches channels produced by the selected MOABB dataset/paradigm.
- Tune `paradigm_kwargs` for windowing and resampling.

## Add a new dataset by implementing a loader class

Use this path when your raw/processed format is not covered by existing loaders.

## 1) Implement a `torch.utils.data.Dataset`

Create file (example): `src/downstream_tasks/dataloader_mydata.py`.

Recommended skeleton:

```python
from torch.utils.data import Dataset
import torch

from downstream_tasks.position_utils import load_positions


class MyDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None, scale_factor=1.0):
        self.path = path
        self.mode = mode
        self.scale_factor = scale_factor

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()
        self.samples = ...  # build list of sample descriptors for this split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = ...  # load single sample + label
        return {
            "sample": torch.as_tensor(x).float() / self.scale_factor,
            "label": torch.as_tensor(y).long(),
        }

    def collate(self, batch):
        x = torch.stack([b["sample"] for b in batch])
        y = torch.stack([b["label"] for b in batch]).view(-1)
        pos = self.positions.unsqueeze(0).repeat(len(batch), 1, 1)
        return {"sample": x, "label": y, "pos": pos}
```

## 2) Point task config to your new `_target_`

```yaml
data_loader:
  dataset:
    _target_: downstream_tasks.dataloader_mydata.MyDataset
    path: ${data_root}/my_data
    electrodes: ["Fz", "Cz", "Pz", ...]
    scale_factor: 100.0
  batch_size: 64
  seed: ${seed}
  splits: ["train", "val", "test"]
```

## 3) Validate compatibility

Checklist:
1. `batch["sample"]` shape is consistent with `task.n_chans` and `task.duration`.
2. `batch["label"]` is integer class indices in `[0, n_classes-1]`.
3. `batch["pos"]` shape is `[B, C, 3]` and channel order matches data channels.
4. A 1-2 epoch `training_mode=lp` smoke test runs end-to-end.

## Common pitfalls

- Mismatch between `electrodes` order and actual channel order in `sample`.
- Wrong label encoding (1-based labels instead of 0-based).
- Returning scalar labels with inconsistent shape across samples.
- Not providing `pos` in `collate`.
- `task.duration` or `task.n_chans` not matching actual sample tensors.

## Related docs

- Config system: `src/configs/README.md`
- Task configs: `src/configs/task/`
- Preprocessing scripts: `preprocessing/README.md`
