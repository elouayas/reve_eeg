"""Utilities to resolve electrode names or files into 3D position tensors."""

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel


@lru_cache(maxsize=1)
def _get_position_model():
    return AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
        dtype="auto",
        cache_dir=".cache",
    )


def load_positions(positions_path=None, electrode_names=None):
    """
    Loads electrode positions.

    Args:
        positions_path (str, optional): Path to .npy file containing positions.
        electrode_names (list[str], optional): List of electrode names.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing 3D coordinates.
    """

    if electrode_names is not None and len(electrode_names) > 0:
        print("Loading from electrode names")
        model = _get_position_model()

        # Handle bipolar montages: 'FP1-F7' -> average of FP1 and F7
        processed_names = []
        is_bipolar = []
        for name in electrode_names:
            if "-" in name:
                processed_names.extend(name.split("-"))
                is_bipolar.append(True)
            else:
                processed_names.append(name)
                is_bipolar.append(False)

        with torch.no_grad():
            all_positions = model(processed_names).float().cpu()

        if not any(is_bipolar):
            return all_positions

        # Reconstruct positions with averaging
        final_positions = []
        ptr = 0
        for bipolar in is_bipolar:
            if bipolar:
                # Average of the next two positions
                avg_pos = (all_positions[ptr] + all_positions[ptr + 1]) / 2.0
                final_positions.append(avg_pos)
                ptr += 2
            else:
                final_positions.append(all_positions[ptr])
                ptr += 1

        return torch.stack(final_positions)

    if positions_path is not None:
        print("Loading from positions path")
        try:
            positions_ = np.load(positions_path, allow_pickle=True)
            return torch.from_numpy(positions_).float()
        except Exception as e:
            print(f"Failed to load positions from {positions_path}: {e}")
            raise e

    raise ValueError("Either 'electrode_names' or 'positions_path' must be provided.")
