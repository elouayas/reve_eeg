"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
import os
import pickle

import lmdb
import mne
from einops import rearrange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stress dataset")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the stress dataset",
    )
    parser.add_argument(
        "--processed",
        type=str,
        required=True,
        help="Directory to save the processed data",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["reve", "cbramod"],
        default="reve",
        help="Type of preprocessing to apply",
    )

    args = parser.parse_args()

    root_dir = args.root
    processed_dir = args.processed

    files = [file for file in os.listdir(root_dir) if file.endswith(".edf")]
    files = sorted(files)
    print(files)

    files_dict = {
        "train": files[:56],
        "val": files[56:64],
        "test": files[64:],
    }
    print(files_dict)
    dataset = {
        "train": [],
        "val": [],
        "test": [],
    }

    selected_channels = [
        "EEG Fp1",
        "EEG Fp2",
        "EEG F3",
        "EEG F4",
        "EEG F7",
        "EEG F8",
        "EEG T3",
        "EEG T4",
        "EEG C3",
        "EEG C4",
        "EEG T5",
        "EEG T6",
        "EEG P3",
        "EEG P4",
        "EEG O1",
        "EEG O2",
        "EEG Fz",
        "EEG Cz",
        "EEG Pz",
        "EEG A2-A1",
    ]

    os.makedirs(processed_dir, exist_ok=True)
    db = lmdb.open(processed_dir, map_size=1_000_000_000)

    for files_key, files_list in files_dict.items():
        for file in files_list:
            raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True)
            raw.pick(selected_channels)
            raw.reorder_channels(selected_channels)
            raw.resample(200)

            eeg = raw.get_data(units="uV")
            chs, points = eeg.shape
            a = points % (5 * 200)
            if a != 0:
                eeg = eeg[:, :-a]
            eeg = eeg.reshape(20, -1, 5, 200).transpose(1, 0, 2, 3)

            if args.type == "reve":
                eeg = rearrange(eeg, "b c t d -> b c (t d)", d=200)
            # else do nothing

            label = int(file[-5])

            for i, sample in enumerate(eeg):
                sample_key = f"{file[:-4]}-{i}"
                # print(sample_key)
                data_dict = {"sample": sample, "label": label - 1}
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()
