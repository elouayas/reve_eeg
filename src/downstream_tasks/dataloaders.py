"""
Common dataloaders for different tasks.
"""

import os
import pickle
from os.path import join as pjoin

import hydra
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from downstream_tasks.position_utils import load_positions


################## Datasets ##################


class LMDBDataset(Dataset):
    def __init__(self, path, positions=None, electrodes=None, mode="train", scale_factor=100):
        super(LMDBDataset, self).__init__()
        self.path = path
        self.scale_factor = scale_factor

        # Open environment briefly to load keys, then close it to avoid issues with multi-processing
        env = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)
        with env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))[mode]  # type: ignore
        env.close()

        if positions is not None:
            positions = pjoin(path, positions)

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)
        self.db = None

    def __len__(self):
        return len((self.keys))

    def _init_db(self):
        if self.db is None:
            self.db = lmdb.open(self.path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)

    def __getitem__(self, index):
        self._init_db()
        assert self.db is not None, "LMDB environment not initialized"
        key = self.keys[index]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair["sample"]
        label = pair["label"]

        ret = {
            "sample": data / self.scale_factor,
            "label": label,
        }
        return ret

    def _to_tensor(self, data):
        return torch.from_numpy(data).float()

    def collate(self, batch):
        x_data = np.array([x["sample"] for x in batch])
        y_label = np.array([x["label"] for x in batch])
        N = len(batch)
        positions = self.positions.repeat(N, 1, 1)
        return {
            "sample": self._to_tensor(x_data),
            "label": self._to_tensor(y_label).long(),
            "pos": positions,
        }


class NeuroLMDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None, scale_factor=100):
        super(NeuroLMDataset, self).__init__()

        self.scale_factor = scale_factor
        self.path = pjoin(path, mode)
        ls = [f for f in os.listdir(self.path) if f.endswith(".pkl")]
        ls = [pjoin(self.path, f) for f in ls]
        self.files = sorted(ls)

        print(f"Found {len(self.files)} files in {self.path}")

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

    def __len__(self):
        return len(self.files)

    def _to_tensor(self, data):
        return torch.from_numpy(data).float()

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            sample = pickle.load(f)
        X = sample["X"]
        Y = int(sample["y"])

        return {
            "sample": self._to_tensor(X / self.scale_factor),
            "label": torch.tensor(Y).long().unsqueeze(0),
            "pos": self.positions,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.tensor([x["label"] for x in batch]),
            "pos": self.positions.repeat(len(batch), 1, 1),
        }


###############################################################################################


def get_data_loaders(config, loader_config, rank=None) -> dict[str, DataLoader]:
    """
    Get data loaders for training, validation, and testing.
    Args:
        config: Configuration object containing dataset and batch size.
    Returns:
        dict: Dictionary containing data loaders for train, val, and test.
    """

    splits = config.get("splits", ["train", "val", "test"])

    train_dataset = hydra.utils.instantiate(config.dataset, mode=splits[0])
    val_dataset = hydra.utils.instantiate(config.dataset, mode=splits[1])
    test_dataset = hydra.utils.instantiate(config.dataset, mode=splits[2])

    if rank is None or rank == 0:
        print(f"Train: {len(train_dataset):,} | Valid: {len(val_dataset):,} | Test: {len(test_dataset):,}")
        print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")

    train_sampler = None
    if rank is not None:
        import idr_torch  # noqa: PLC0415

        print("Using distributed sampler", rank, idr_torch.size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            rank=rank,
            num_replicas=idr_torch.size,
        )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=train_dataset.collate,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            sampler=train_sampler,
            **loader_config,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=val_dataset.collate,
            shuffle=False,
            **loader_config,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            collate_fn=test_dataset.collate,
            shuffle=False,
            **loader_config,
        ),
    }
