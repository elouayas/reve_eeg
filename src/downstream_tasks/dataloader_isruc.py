"""ISRUC dataset loader with split construction and sequence/label pairing."""

import os
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from downstream_tasks.position_utils import load_positions


N_THREADS = min(16, os.cpu_count() - 1)  # type: ignore

#######################################################################


class ISRUCDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None):
        super().__init__()
        self.path = path
        self.mode = mode

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

        self.seq_dir = os.path.join(self.path, "seq")
        self.label_dir = os.path.join(self.path, "labels")
        total_pairs = _load_path(self.seq_dir, self.label_dir)
        seqs_labels_path_pair = _split(total_pairs)[mode]

        total_data = []
        total_labels = []

        with Pool(N_THREADS) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        self._process_file_pair,
                        list(seqs_labels_path_pair),
                    ),
                    total=len(seqs_labels_path_pair),
                    desc=self.mode,
                ),
            )

        for seqs, labels in results:
            total_data.extend(seqs)
            total_labels.extend(labels)

        self.seqs_labels_path_pair = [(data, label) for data, label in zip(total_data, total_labels)]

        print(self.seqs_labels_path_pair[0])

    def _process_file_pair(self, args):
        seq_path, label_path = args
        seqs = np.load(seq_path)
        labels = np.load(label_path)
        assert seqs.shape[0] == labels.shape[0], f"seqs: {seqs.shape}, labels: {labels.shape}"
        return [(seq_path, i) for i in range(seqs.shape[0])], [(label_path, i) for i in range(labels.shape[0])]

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, index):
        seq_path, seq_idx = self.seqs_labels_path_pair[index][0]
        label_path, label_idx = self.seqs_labels_path_pair[index][1]

        seq = np.load(seq_path)[seq_idx]
        label = np.load(label_path)[label_idx]

        return {
            "sample": self._to_tensor(seq) / 10.0,
            "label": torch.tensor(label).long(),
            "pos": self.positions,
        }

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


def _load_path(seq_dir, label_dir):
    seqs_labels_path_pair = []
    subject_dirs_seq = []
    subject_dirs_labels = []
    for subject_num in range(1, 101):
        subject_dirs_seq.append(os.path.join(seq_dir, f"ISRUC-group1-{subject_num}"))
        subject_dirs_labels.append(os.path.join(label_dir, f"ISRUC-group1-{subject_num}"))

    for subject_seq, subject_label in zip(subject_dirs_seq, subject_dirs_labels):
        subject_pairs = []
        seq_fnames = os.listdir(subject_seq)
        label_fnames = os.listdir(subject_label)
        for seq_fname, label_fname in zip(seq_fnames, label_fnames):
            subject_pairs.append((os.path.join(subject_seq, seq_fname), os.path.join(subject_label, label_fname)))
        seqs_labels_path_pair.append(subject_pairs)
    return seqs_labels_path_pair


def _split(seqs_labels_path_pair):
    SPLIT1 = 80
    SPLIT2 = 90

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for i in range(100):
        if i < SPLIT1:
            train_pairs.extend(seqs_labels_path_pair[i])
        elif i < SPLIT2:
            val_pairs.extend(seqs_labels_path_pair[i])
        else:
            test_pairs.extend(seqs_labels_path_pair[i])

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}
