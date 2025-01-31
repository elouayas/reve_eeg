"""TUH (TUAB/TUEV) dataset loaders with train/val/test split handling."""

import os
import pickle
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class TUEV(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
        positions=None,
        electrodes=None,
        scale_factor: float = 1e4,
    ):
        self.scale_factor = scale_factor

        assert mode in ["train", "val", "test"]
        if mode == "test":
            self.data_path = pjoin(path, "processed_eval")
            self.file_names = sorted([f for f in os.listdir(self.data_path) if not f.startswith(".")])
        elif mode in {"train", "val"}:
            self.data_path = pjoin(path, "processed_train")
            all_files = sorted([f for f in os.listdir(self.data_path) if not f.startswith(".")])

            rng = np.random.default_rng(seed=42)
            rng.shuffle(all_files)

            n_train = int(0.8 * len(all_files))
            if mode == "train":
                self.file_names = all_files[:n_train]
            else:
                self.file_names = all_files[n_train:]

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        X = torch.tensor(data["signal"])
        Y = torch.tensor(data["label"]).squeeze(-1) - 1

        return {
            "sample": X.float() * self.scale_factor,
            "label": Y.long(),
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "pos": self.positions.unsqueeze(0).repeat(len(batch), 1, 1),
        }


class TUAB(Dataset):
    N_TRIALS = 409_455
    N_TIMES = 2_000
    N_CHANS = 21

    DATASET_NAME = "TUAB"
    SEED = 42

    def __init__(
        self,
        path: str,
        mode: str = "train",
        positions=None,
        electrodes=None,
        scale_factor: float = 1e4,
    ):
        self.scale_factor = scale_factor

        # Splitting logic
        test_start, test_end = (0, 36944)
        train_val_start, train_val_end = (36945, 409454)

        if mode == "test":
            self.segs = np.arange(test_start, test_end + 1).tolist()
        else:
            train_val_indices = np.arange(train_val_start, train_val_end + 1)
            rng = np.random.default_rng(seed=self.SEED)
            rng.shuffle(train_val_indices)
            n_train = int(0.8 * len(train_val_indices))
            if mode == "train":
                self.segs = train_val_indices[:n_train].tolist()
            else:  # val
                self.segs = train_val_indices[n_train:].tolist()

        # Load Memmaps
        self.eeg_files = np.memmap(
            pjoin(path, "TUAB", f"X_-_eeg_-_{self.DATASET_NAME}.npy"),
            mode="r",
            shape=(self.N_TRIALS, self.N_CHANS, self.N_TIMES),
            dtype="float32",
        )
        self.y_files = np.memmap(
            pjoin(path, "TUAB", f"Y_-_eeg_-_{self.DATASET_NAME}.npy"),
            mode="r",
            shape=self.N_TRIALS,
            dtype="int64",
        )

        # Load positions
        if positions is None:
            pos_path = pjoin(path, "TUAB", f"pos_-_eeg_-_{self.DATASET_NAME}.npy")
            if os.path.exists(pos_path):
                positions = pos_path

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()

    def __len__(self):
        return len(self.segs)

    def __getitem__(self, index):
        n_trial = self.segs[index]

        eeg = self.eeg_files[n_trial].copy()
        target = self.y_files[n_trial].copy()

        # Convert to tensor
        eeg = torch.from_numpy(eeg).float()
        target = torch.tensor(target).long()

        return {
            "sample": eeg * self.scale_factor,
            "label": target,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "pos": self.positions.unsqueeze(0).repeat(len(batch), 1, 1),
        }
