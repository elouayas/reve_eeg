# Preprocessing Guide

This folder contains dataset-specific preprocessing scripts used to convert raw EEG corpora into formats consumed by downstream loaders in `src/downstream_tasks`.

## Scope

The scripts are adapted from:
- [CBraMod](https://github.com/wjq-learning/CBraMod)
- [NeuroLM](https://github.com/935963004/NeuroLM)

Most scripts produce either:
- LMDB datasets with serialized samples (`{"sample": ..., "label": ...}`) and a `__keys__` split index, or
- file-based sequence/label arrays (ISRUC, HMC variants)

## General conventions

- Target sampling rate is generally `200 Hz`.
- Labels are mapped to integer class IDs.
- Splits are usually materialized as `train` / `val` / `test`.
- Some scripts support `--type reve|cbramod` to choose output tensor layout.

## Output formats

### LMDB-based format
Used by scripts like BCICIV2a, FACED, MUMTAZ, PhysioNet, Speech, Stress.

Each LMDB key stores a pickled dict:
- `sample`: EEG tensor/array
- `label`: integer class label

A special LMDB key `__keys__` stores split membership:
- `{"train": [...], "val": [...], "test": [...]}`

This format is consumed by `downstream_tasks.dataloaders.LMDBDataset`.

### Directory/file-based format
- `preprocessing_hmc.py` writes `.pkl` files in `train/`, `eval/`, `test/`.
- `ISRUC/prepare_ISRUC.py` writes numpy sequences and labels in `seq/` and `labels/` trees.

These are consumed by dedicated loaders (`NeuroLMDataset`, `ISRUCDataset`).

## Script reference

### BCIC IV-2a
File: `preprocessing/preprocessing_bciciv2a.py`

Arguments:
- `--raw` (required): directory with `.gdf` files
- `--processed` (required): output LMDB directory
- `--type` (optional): `reve` or `cbramod` (default `reve`)

Example:
```bash
python preprocessing/preprocessing_bciciv2a.py \
  --raw /path/to/BCIC-IV-2a/raw \
  --processed /path/to/BCIC-IV-2a/processed \
  --type reve
```

### FACED
File: `preprocessing/preprocessing_faced.py`

Arguments:
- `--root` (required): root directory containing FACED pickle files
- `--processed` (required): output directory (script writes to a `processed_cbramod` variant path)

Example:
```bash
python preprocessing/preprocessing_faced.py \
  --root /path/to/FACED/raw \
  --processed /path/to/FACED/processed
```

### HMC
File: `preprocessing/preprocessing_hmc.py`

Arguments:
- `--raw` (required): root directory containing EDF + sleep scoring files
- `--processed` (required): output directory

Example:
```bash
python preprocessing/preprocessing_hmc.py \
  --raw /path/to/HMC/raw \
  --processed /path/to/HMC/processed
```

Output:
- `processed/train/*.pkl`
- `processed/eval/*.pkl`
- `processed/test/*.pkl`

### MUMTAZ
File: `preprocessing/preprocessing_mumtaz.py`

Arguments:
- `--root` (required): root directory with `.edf`
- `--processed` (required): output LMDB directory
- `--type` (optional): `reve` or `cbramod` (default `reve`)

Example:
```bash
python preprocessing/preprocessing_mumtaz.py \
  --root /path/to/MUMTAZ/raw \
  --processed /path/to/MUMTAZ/processed \
  --type reve
```

### PhysioNet MI
File: `preprocessing/preprocessing_physio.py`

Arguments:
- `--root` (required): PhysioNet subject root
- `--processed` (required): output LMDB directory
- `--type` (optional): `reve` or `cbramod` (default `reve`)

Example:
```bash
python preprocessing/preprocessing_physio.py \
  --root /path/to/PhysioNet/raw \
  --processed /path/to/PhysioNet-MI/processed \
  --type reve
```

### Speech (BCIC2020)
File: `preprocessing/preprocessing_speech.py`

Arguments:
- `--train` (required): train `.mat` directory
- `--val` (required): validation `.mat` directory
- `--test` (required): test files directory
- `--excel` (required): label spreadsheet path
- `--processed` (required): output LMDB directory

Example:
```bash
python preprocessing/preprocessing_speech.py \
  --train /path/to/BCIC2020/train \
  --val /path/to/BCIC2020/val \
  --test /path/to/BCIC2020/test \
  --excel /path/to/BCIC2020/labels.xlsx \
  --processed /path/to/BCIC2020/processed
```

### Stress (Mental Arithmetic)
File: `preprocessing/preprocessing_stress.py`

Arguments:
- `--root` (required): root directory with EDF files
- `--processed` (required): output LMDB directory
- `--type` (optional): `reve` or `cbramod` (default `reve`)

Example:
```bash
python preprocessing/preprocessing_stress.py \
  --root /path/to/mental_arithmetic/raw \
  --processed /path/to/mental_arithmetic/processed \
  --type reve
```

### ISRUC
File: `preprocessing/ISRUC/prepare_ISRUC.py`

Arguments:
- `--raw` (required): root ISRUC group directory
- `--processed` (required): output template with `{}` placeholder

Example:
```bash
python preprocessing/ISRUC/prepare_ISRUC.py \
  --raw /path/to/ISRUC/group1 \
  --processed /path/to/ISRUC/processed/{}
```

Important:
- The `--processed` argument is formatted as a template in code (`.format("seq")`, `.format("labels")`).
- Use a template such as `/path/to/ISRUC/processed/{}` (no space between braces).

Expected output:
- `/path/to/ISRUC/processed/seq/...`
- `/path/to/ISRUC/processed/labels/...`

## Mapping to downstream task configs

After preprocessing, set `data_root` so task configs in `src/configs/task/*.yaml` can resolve dataset paths.

Examples:
- BCICIV2a expects `${data_root}/BCIC-IV-2a`
- FACED expects `${data_root}/FACED/processed`
- HMC expects `${data_root}/HMC/processed/`
- ISRUC expects `${data_root}/ISRUC/processed`
- MUMTAZ expects `${data_root}/MUMTAZ/processed`
- Physio expects `${data_root}/PhysioNet-MI/processed`
- Speech expects `${data_root}/BCIC2020/processed`
- Stress expects `${data_root}/mental_arithmetic/processed`

## Quick validation checklist

1. Confirm output path exists and is non-empty.
2. For LMDB outputs, verify `__keys__` exists and contains expected splits.
3. Run a short downstream smoke test with `task=<dataset>` and `training_mode=lp`.
4. Check sample shape and class cardinality against task config (`n_chans`, `n_classes`, `duration`).
