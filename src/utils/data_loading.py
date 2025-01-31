"""
Data loading utilities for EEG pre-training.

This module reads three CSV metadata files located in `<data_path>/csv_recordings/`:

1. **df_big.csv** — One row per "big recording" (an aggregated EEG data file).
   Columns: big_recording_index, duration, n_chans, chans_names, config

2. **df_corrected.csv** — One row per session/segment within a big recording.
   Columns: file, big_recording_index, index, duration, n_chans, mult, dataset,
            class, flag_remove, flag_reduce, n_chans_to_remove
   - `flag_remove`: "True" if the segment should be excluded entirely.
   - `flag_reduce`: list of channel indices to drop (parsed via ast.literal_eval).
   - `n_chans_to_remove`: number of channels flagged for removal.

3. **df_stats_tmp.csv** — One row per big recording, gives shape info for
   per-session normalization statistics (mean/std stored as .npy memmaps).
   Columns: big_recording_index, n_sessions, n_chans

The pipeline:
    CSVs  →  filter by recording subset  →  compute_group_segments (windowing)
          →  EEGDataset (memmap reads + normalization + masking)
          →  GroupedSampler (channel-homogeneous batches)  →  DataLoader
"""

import ast
import csv
import gc
import random
from collections import defaultdict
from os.path import join as pjoin

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, Dataset, Sampler


SUBSET_VAL = [78, 83, 84, 93, 97]
SUBSET_TRAIN = [72, 73, 74, 75, 77, 79, 80, 81, 82, 85, 86, 87, 88, 90, 92, 98, 99, 100, 101, 102, 103, 104]


OPEN_VAL = [
    227,
    113,
    137,
    178,
    110,
    90,
    358,
    160,
    156,
    233,
    131,
    81,
    394,
    229,
    312,
    401,
    437,
    267,
    116,
    132,
    189,
    108,
    425,
    143,
    274,
    101,
    163,
    382,
    188,
    246,
    167,
    176,
    411,
    364,
    369,
]
OPEN_TRAIN = [
    196,
    285,
    155,
    219,
    127,
    228,
    121,
    327,
    268,
    82,
    145,
    105,
    194,
    435,
    192,
    262,
    436,
    199,
    314,
    209,
    191,
    264,
    336,
    238,
    165,
    204,
    169,
    277,
    341,
    118,
    211,
    242,
    236,
    173,
    370,
    157,
    320,
    139,
    144,
    100,
    372,
    168,
    222,
    249,
    359,
    423,
    291,
    74,
    255,
    421,
    102,
    198,
    346,
    273,
    259,
    230,
    225,
    210,
    182,
    431,
    440,
    260,
    135,
    405,
    308,
    331,
    154,
    381,
    360,
    433,
    403,
    337,
    345,
    218,
    356,
    424,
    279,
    184,
    86,
    166,
    324,
    283,
    332,
    241,
    388,
    393,
    133,
    250,
    353,
    195,
    92,
    357,
    335,
    288,
    354,
    343,
    319,
    248,
    158,
    347,
    328,
    119,
    396,
    392,
    111,
    99,
    226,
    243,
    323,
    114,
    293,
    384,
    318,
    208,
    371,
    282,
    439,
    434,
    206,
    402,
    231,
    190,
    294,
    303,
    128,
    342,
    149,
    123,
    266,
    186,
    115,
    325,
    216,
    426,
    409,
    410,
    350,
    124,
    181,
    313,
    428,
    134,
    276,
    315,
    93,
    106,
    244,
    306,
    284,
    301,
    172,
    232,
    316,
    98,
    321,
    290,
    207,
    263,
    322,
    212,
    374,
    275,
    256,
    140,
    220,
    202,
    366,
    297,
    122,
    334,
    286,
    142,
    161,
    417,
    387,
    389,
    257,
    441,
    429,
    187,
    367,
    432,
    223,
    271,
    351,
    305,
    270,
    298,
    146,
    193,
    295,
    339,
    365,
    215,
    117,
    252,
    153,
    307,
    197,
    174,
    280,
    412,
    177,
    300,
    136,
    129,
    400,
    247,
    407,
    180,
    368,
    329,
    265,
    287,
    378,
    185,
    406,
    383,
    234,
    415,
    309,
    352,
    344,
    361,
    237,
    427,
    258,
    138,
    278,
    304,
    170,
    380,
    375,
    213,
    330,
    413,
    385,
    87,
    422,
    125,
    420,
    338,
    292,
    107,
    253,
    373,
    269,
    386,
    254,
    281,
    109,
    355,
    200,
    416,
    349,
    438,
    80,
    289,
    104,
    296,
    217,
    239,
    245,
    224,
    390,
    377,
    79,
    404,
    310,
    397,
    362,
    333,
    395,
    317,
    340,
    203,
    159,
    391,
    299,
    151,
    85,
    83,
    348,
    214,
    130,
    251,
    221,
    399,
    205,
    162,
    442,
    126,
    179,
    414,
    112,
    201,
    376,
    272,
    408,
    302,
    175,
    326,
    148,
    120,
    430,
    183,
    147,
    171,
    418,
    235,
    261,
    164,
    311,
    88,
    141,
    398,
    419,
    89,
    152,
    363,
    240,
    103,
    150,
]


def _read_csv(path: str) -> list[dict]:
    """Read a single CSV file and return its rows as a list of dicts."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f, delimiter=","))


def _filter_by_recording_set(rows: list[dict], recording_set: set | list) -> list[dict]:
    """Keep only rows whose big_recording_index is in *recording_set*."""
    allowed = set(recording_set)
    return [r for r in rows if int(r["big_recording_index"]) in allowed]


def _group_by(rows: list[dict], key: str) -> dict[int, list[dict]]:
    """Group *rows* by the integer value of *key*."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row[key])].append(row)
    return dict(groups)


def _make_window_keys(  # noqa: PLR0913
    big_rec_index: str,
    session_idx: int,
    start: int,
    end: int,
    window_duration: int,
    chans_to_remove: str | None = None,
) -> list[str]:
    offsets = np.arange(start, end - window_duration, window_duration)
    if chans_to_remove is None:
        return [f"{big_rec_index}_-_{session_idx}_-_{w}" for w in offsets]
    return [f"{big_rec_index}_-_{session_idx}_-_{w}_-_{chans_to_remove}" for w in offsets]


def compute_group_segments(data, data_big, window_duration):
    """Compute channel-grouped window segment keys from CSV metadata.

    Returns
    -------
    dict_groups : dict[int, list[str]]
        Mapping from effective channel count to list of window keys.
    segments : list[str]
        Flat list of all window keys.
    """
    chan_aggregates = _group_by(data_big, "n_chans")
    dict_groups: dict[int, list[str]] = defaultdict(list)
    dict_groups_incorrect: dict[int, list[str]] = defaultdict(list)

    for n_chans, recordings in chan_aggregates.items():
        windows: list[str] = []
        for recording_data in recordings:
            b_idx = recording_data["big_recording_index"]
            index_data = [d for d in data if d["big_recording_index"] == b_idx]

            # Flag faulty reduce for removal
            if index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:
                index_data[-1]["flag_remove"] = "True"

            # Compute cumulative start/end for each session
            corrected_starts = [0] + np.cumsum([int(d["duration"]) for d in index_data[:-1]]).tolist()
            for d, s in zip(index_data, corrected_starts):
                d["start"] = s
                d["end"] = s + int(d["duration"])

            # Filter sessions shorter than one window
            index_data = [d for d in index_data if int(d["duration"]) > window_duration]

            # Create window keys for each session
            for i, d in enumerate(index_data):
                if d["flag_remove"] == "True":
                    continue

                n_to_remove = int(d["n_chans_to_remove"])
                if n_to_remove == 0:
                    windows += _make_window_keys(b_idx, i, d["start"], d["end"], window_duration)
                else:
                    chans_str = "/".join(str(x) for x in ast.literal_eval(str(d["flag_reduce"])))
                    keys = _make_window_keys(b_idx, i, d["start"], d["end"], window_duration, chans_str)
                    dict_groups_incorrect[n_chans - n_to_remove] += keys

        dict_groups[n_chans] += windows

    # Merge faulty-channel windows into the correct effective-channel groups
    for effective_chans, keys in dict_groups_incorrect.items():
        dict_groups[effective_chans] += keys

    segments = [key for keys in dict_groups.values() for key in keys]
    dict_groups = dict(sorted(dict_groups.items()))

    return dict_groups, segments


def spatial_masking(
    C,
    masking_ratio,
    radius,
    precomp_masked_indices=None,
):
    n_channels = C.shape[0]
    n_masked_channels = int(masking_ratio * n_channels)
    mask = np.zeros(n_channels, dtype=bool)

    # Pre-add masks from dropout
    if precomp_masked_indices is not None:
        mask[precomp_masked_indices] = True

    # Use a KD-Tree to find neighboring channels
    kdtree = KDTree(C)

    unmasked_indices = np.where(~mask)[0]
    while len(unmasked_indices) > 0 and np.sum(mask) < n_masked_channels:
        unmasked_indices = np.where(~mask)[0]
        _index = np.random.choice(unmasked_indices)
        _channel = C[_index]
        _neighbors = kdtree.query_ball_point(_channel, radius)
        mask[_neighbors] = True
    masked_indices = np.where(mask)[0]

    return masked_indices


def create_block_masks(  # noqa: PLR0913
    n_chans,
    masking_ratio,
    radius_spat_mask,
    radius_temp_mask,
    num_patches,
    pos,
    dropout_ratio,
    dropout_ratio_radius,
):
    num_block_patches = num_patches // radius_temp_mask
    num_isolated_patches = num_patches % radius_temp_mask
    num_masked_chans = int(masking_ratio * n_chans)
    chan_indices = np.arange(n_chans)

    # Obtain dropout masks in first pass
    if dropout_ratio > 0:
        dropout_channels = spatial_masking(pos, dropout_ratio, dropout_ratio_radius)[: int(dropout_ratio * n_chans)]
    else:
        dropout_channels = None

    # Create block masks
    block_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask, dropout_channels)[:num_masked_chans]
        for _ in range(num_block_patches)
    ]
    idx_block = np.repeat(
        np.array(block_masks)[:, np.newaxis, :],
        radius_temp_mask,
        axis=1,
    ).reshape(-1, num_masked_chans)

    # Create masks for remaining patches (case where num of patches and masks don't overlap)
    isolated_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask)[:num_masked_chans] for _ in range(num_isolated_patches)
    ]
    idx_isolated = np.array(isolated_masks)

    mask = np.concatenate([idx_block, idx_isolated], axis=0)

    # Efficient unmasks generation
    unmask = np.array([np.setdiff1d(chan_indices, masked_indices, assume_unique=True) for masked_indices in mask])

    flat_indices = np.arange(mask.shape[0])[:, np.newaxis]
    masked_indices = (n_chans * flat_indices + mask).ravel()
    unmasked_indices = (n_chans * flat_indices + unmask).ravel()
    np.random.shuffle(masked_indices)
    np.random.shuffle(unmasked_indices)

    batch_mask = torch.from_numpy(masked_indices)
    batch_unmask = torch.from_numpy(unmasked_indices)
    return batch_mask, batch_unmask


class EEGDataset(Dataset):
    def __init__(  # noqa: PLR0913
        self,
        segments,
        groups,
        data_big,
        data_stats,
        recordings_path,
        window_duration,
        clip,
        block_masking,
        masking_window,
        masking_overlap,
        masking_ratio,
        radius_spat_mask,
        radius_temp_mask,
        dropout_ratio,
        dropout_radius,
        no_masking=False,
        manual_seed=False,
    ):
        self.path = recordings_path
        self.path_pos = "/".join(self.path.split("/")[:-1] + ["positions"])
        self.path_stats = "/".join(self.path.split("/")[:-1] + ["stats"])

        self.segments = segments
        self.groups = groups
        self.window_duration = window_duration
        self.clip = clip
        self.block_masking = block_masking
        self.masking_window = masking_window
        self.masking_overlap = masking_overlap
        self.masking_ratio = masking_ratio
        self.no_masking = no_masking
        if self.block_masking:
            self.radius_spat_mask = radius_spat_mask
            self.radius_temp_mask = radius_temp_mask
            self.dropout_ratio = dropout_ratio
            self.dropout_radius = dropout_radius
        else:
            self.manual_seed = manual_seed

        self.data_big = data_big
        self.data_stats = data_stats
        self.init_files_pos()

        self.counter = 0
        self.max_counter = 500000

    def init_files_pos(self):
        self.files = {}
        self.positions_cache = {}
        for recording_data in self.data_big:
            r_i = int(recording_data["big_recording_index"])
            r_t = int(recording_data["duration"])
            r_c = int(recording_data["n_chans"])

            memmap_array = np.memmap(
                pjoin(self.path, f"recording_-_eeg_-_{r_i}.npy"),
                mode="r",
                shape=(r_t, r_c),
                dtype="float32",
            )

            self.files[r_i] = memmap_array
            self.positions_cache[r_i] = np.load(
                pjoin(self.path_pos, f"recording_-_positions_-_{r_i}.npy"),
            )

        self.stats = {}
        for recording_data in self.data_stats:
            r_i = int(recording_data["big_recording_index"])
            n_s = int(recording_data["n_sessions"])
            n_c = int(recording_data["n_chans"])

            memmap_stats = np.memmap(
                pjoin(self.path_stats, f"recording_-_stats_-_{r_i}.npy"),
                mode="r",
                shape=(n_s, 2, n_c),
                dtype="float32",
            )

            self.stats[r_i] = memmap_stats

    def del_files(self):
        for k, v in self.files.items():
            v._mmap.close()
            del v

        for k, v in self.stats.items():
            v._mmap.close()
            del v

        del self.files, self.positions_cache, self.stats
        gc.collect()

    def __getitem__(self, index):
        if self.counter >= 500000:
            self.del_files()
            self.init_files_pos()
            self.counter = 0
        self.counter += 1

        index_ = index.split("_-_")
        if len(index_) == 3:
            b_rec, rec, offset = index_
            faulty_chans = False
        elif len(index_) == 4:
            b_rec, rec, offset, faulty_chans = index_
            faulty_chans = [int(x) for x in faulty_chans.split("/")]
        else:
            raise ValueError(f"Invalid index format: {index}")
        b_rec, rec, offset = int(b_rec), int(rec), int(offset)

        positions = self.positions_cache[b_rec].copy()
        stats = self.stats[b_rec][rec]
        eeg = self.files[b_rec][offset : offset + self.window_duration].copy()

        # Mask faulty channels if applicable
        if faulty_chans:
            mask = np.ones(positions.shape[0], dtype=bool)
            mask[faulty_chans] = False
            eeg = eeg[:, mask]
            positions = positions[mask]
            stats = stats[:, mask]

        # Normalization
        eps = 1e-10 if any(stats[1] == 0) else 0
        eeg -= stats[0]
        eeg /= stats[1] + eps
        eeg = torch.from_numpy(eeg)
        eeg = eeg.transpose(0, 1)

        # Case one only clip, no masking
        if self.no_masking:
            return eeg.float().clip(-self.clip, self.clip), torch.from_numpy(positions).float()

        # Case two block masking
        elif self.block_masking:
            patches = eeg.unfold(dimension=1, size=self.masking_window, step=self.masking_window - self.masking_overlap)
            c, h, p = patches.shape
            batch_mask, batch_unmask = create_block_masks(
                c,
                self.masking_ratio,
                self.radius_spat_mask,
                self.radius_temp_mask,
                h,
                positions,
                self.dropout_ratio,
                self.dropout_radius,
            )
            return (
                eeg.float().clip(-self.clip, self.clip),
                torch.from_numpy(positions).float(),
                batch_mask,
                batch_unmask,
            )

        # Case three simple masking
        else:
            patches = eeg.unfold(dimension=1, size=self.masking_window, step=self.masking_window - self.masking_overlap)
            c, h, p = patches.shape
            num_patches = c * h
            num_masks = int(self.masking_ratio * num_patches)
            if self.manual_seed is not None and self.manual_seed is not False:
                torch.manual_seed(int(self.manual_seed))
            rand_indices = torch.rand(1, num_patches).argsort(dim=-1)
            batch_mask, batch_unmask = rand_indices[:, :num_masks], rand_indices[:, num_masks:]
            batch_mask, batch_unmask = batch_mask.squeeze(0), batch_unmask.squeeze(0)
            return (
                eeg.float().clip(-self.clip, self.clip),
                torch.from_numpy(positions).float(),
                batch_mask,
                batch_unmask,
            )

    def __len__(self):
        return len(self.segments)


def get_local_batch_size(c, global_batch_size):
    C = 16 * global_batch_size
    local_batch_size = max(1, int(C // c))
    return local_batch_size


class GroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, n_gpu, mode="train"):
        self.dataset = dataset
        self.segments = self.dataset.segments
        self.groups = self.dataset.groups
        self.mode = mode

        self.global_batch_size = batch_size
        self.drop_last = 0 if drop_last else 1
        self.n_gpu = n_gpu

    def __iter__(self):
        indices = []
        n_chans = list(self.groups.keys())

        if self.mode == "train":
            n_chans = [c for c in n_chans if c > 6]
            for c in n_chans:
                self.local_batch_size = get_local_batch_size(c, self.global_batch_size)
                group_indices = self.groups[c]
                random.shuffle(group_indices)
                indices += [
                    group_indices[i * self.local_batch_size : (i + 1) * self.local_batch_size]
                    for i in range(len(group_indices) // self.local_batch_size + self.drop_last)
                ]
            random.shuffle(indices)
        else:
            for c in n_chans:
                group_indices = self.groups[c]
                indices += [
                    group_indices[i * self.global_batch_size : (i + 1) * self.global_batch_size]
                    for i in range(len(group_indices) // self.global_batch_size + self.drop_last)
                ]

        n_leftover_indices = len(indices) % self.n_gpu
        self.indices = indices if n_leftover_indices == 0 else indices[:-n_leftover_indices]

        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _build_loader(args, data, data_big, data_stats, recordings_path, train: bool):  # noqa: PLR0913
    """Build a DataLoader + EEGDataset + GroupedSampler for train or val."""
    dict_groups, segments = compute_group_segments(data, data_big, args.preprocessing.window_duration)

    masking_cfg = args.preprocessing.masking
    dataset = EEGDataset(
        segments,
        dict_groups,
        data_big,
        data_stats,
        recordings_path,
        window_duration=args.preprocessing.window_duration,
        clip=args.preprocessing.clip,
        block_masking=masking_cfg.use_block if train else False,
        masking_window=masking_cfg.masking_window,
        masking_overlap=masking_cfg.masking_overlap,
        masking_ratio=masking_cfg.ratio,
        radius_spat_mask=masking_cfg.radius_spat_mask,
        radius_temp_mask=masking_cfg.radius_temp_mask,
        dropout_ratio=masking_cfg.dropout_ratio,
        dropout_radius=masking_cfg.dropout_radius,
        no_masking=False,
        manual_seed=False if train else args.seed,
    )

    sampler = GroupedSampler(
        dataset,
        batch_size=args.trainer.batch_size,
        drop_last=True,
        n_gpu=args.trainer.n_gpus * args.trainer.n_nodes,
        mode="train" if train else "val",
    )

    sampler.__iter__()
    len_sampler = len(sampler)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=sampler,
        num_workers=args.data.loader.num_workers,
        persistent_workers=False,
        prefetch_factor=args.data.loader.prefetch_factor,
    )

    return loader, len(dataset), len_sampler


def get_train_val_loaders(
    args,
    return_val=True,
) -> tuple[DataLoader, DataLoader | None, int, int | None, int, int | None]:
    """Entry point: build train (and optionally val) data loaders from config."""
    recordings_path = pjoin(args.data.path, "recordings")
    csv_path = pjoin(args.data.path, "csv_recordings")

    # Read CSV metadata
    csv_big = _read_csv(pjoin(csv_path, "df_big.csv"))
    csv_corrected = _read_csv(pjoin(csv_path, "df_corrected.csv"))
    csv_stats = _read_csv(pjoin(csv_path, "df_stats_tmp.csv"))

    # Determine recording sets
    if args.data.subset == "small":
        train_set = SUBSET_TRAIN
        val_set = SUBSET_VAL if return_val else None
    elif args.data.subset == "open":
        train_set = OPEN_TRAIN
        val_set = OPEN_VAL if return_val else None
    elif args.data.subset == "all":
        train_set = list({int(row["big_recording_index"]) for row in csv_big})
        val_set = None
    else:
        raise ValueError(f"Unknown data subset: {args.data.subset}")

    # Build train loader
    train_loader, len_train, len_train_sampler = _build_loader(
        args,
        _filter_by_recording_set(csv_corrected, train_set),
        _filter_by_recording_set(csv_big, train_set),
        _filter_by_recording_set(csv_stats, train_set),
        recordings_path,
        train=True,
    )

    if val_set is None:
        return train_loader, None, len_train, None, len_train_sampler, None

    # Build val loader
    val_loader, len_val, len_val_sampler = _build_loader(
        args,
        _filter_by_recording_set(csv_corrected, val_set),
        _filter_by_recording_set(csv_big, val_set),
        _filter_by_recording_set(csv_stats, val_set),
        recordings_path,
        train=False,
    )
    return train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler
