"""MOABB-backed dataset wrapper used by downstream Hydra task configs."""

import hydra
import torch
from moabb.datasets.base import BaseDataset
from moabb.paradigms import P300, SSVEP, MotorImagery
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class MOABBDataset(Dataset):
    def __init__(  # noqa: PLR0913
        self,
        dataset_kwargs,
        slices: dict[str, tuple[int, int]],
        label_map: dict,
        positions=None,
        electrodes=None,
        mode: str = "train",
        cache_dir: str = ".cache",
        scale_factor: float = 1.0,
        paradigm_kwargs: dict | None = None,
    ):
        dataset: BaseDataset = hydra.utils.instantiate(dataset_kwargs)

        self.mode = mode
        self.scale_factor = scale_factor

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

        assert self.mode in slices
        slice_ = slice(*slices[self.mode])
        subject_list = dataset.subject_list[slice_]

        dataset_paradigm = getattr(dataset, "paradigm", "")
        paradigm_kwargs = paradigm_kwargs or {}
        if dataset_paradigm == "imagery":
            paradigm = MotorImagery(**paradigm_kwargs)
        elif dataset_paradigm == "p300":
            paradigm = P300(**paradigm_kwargs)
        elif dataset_paradigm == "ssvep":
            paradigm = SSVEP(**paradigm_kwargs)
        else:  # default to motor imagery
            paradigm = MotorImagery(**paradigm_kwargs)

        cache_config = {
            "use": True,
            "save_raw": True,
            "save_epochs": True,
            "save_array": True,
            "path": cache_dir,
        }

        X, labels, _metadata = paradigm.get_data(
            dataset=dataset,
            subjects=subject_list,
            cache_config=cache_config,
        )

        mapping = {str(k).strip().lower().replace(" ", "_"): int(v) for k, v in label_map.items()}

        self.X = X
        try:
            self.labels = [mapping[str(label).strip().lower().replace(" ", "_")] for label in labels]
        except KeyError as e:
            available = sorted({str(label) for label in labels})
            raise ValueError(f"Unknown label {e.args[0]!r}. Available labels: {available}") from e

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = torch.from_numpy(self.X[index]).float() / self.scale_factor
        label = torch.tensor(self.labels[index]).long()
        return {
            "sample": sample,
            "label": label,
        }

    def collate(self, batch):
        sample = torch.stack([x["sample"] for x in batch])
        label = torch.stack([x["label"] for x in batch])
        N = len(batch)

        return {
            "sample": sample,
            "label": label,
            "pos": self.positions.repeat(N, 1, 1),
        }
