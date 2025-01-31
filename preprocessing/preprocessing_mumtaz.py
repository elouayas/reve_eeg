"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
import os
import pickle

import lmdb
import mne
from einops import rearrange


# Traverse folders
def iter_files(rootDir):
    # Traverse the root directory
    files_H, files_MDD = [], []
    for file in os.listdir(rootDir):
        if not file.endswith(".edf"):
            continue

        if "TASK" not in file:
            if "MDD" in file:
                files_MDD.append(file)
            else:
                files_H.append(file)
    return files_H, files_MDD


selected_channels = [
    "EEG Fp1-LE",
    "EEG Fp2-LE",
    "EEG F3-LE",
    "EEG F4-LE",
    "EEG C3-LE",
    "EEG C4-LE",
    "EEG P3-LE",
    "EEG P4-LE",
    "EEG O1-LE",
    "EEG O2-LE",
    "EEG F7-LE",
    "EEG F8-LE",
    "EEG T3-LE",
    "EEG T4-LE",
    "EEG T5-LE",
    "EEG T6-LE",
    "EEG Fz-LE",
    "EEG Cz-LE",
    "EEG Pz-LE",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess mumtaz dataset")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--processed",
        type=str,
        required=True,
        help="Directory to save the processed dataset",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["reve", "cbramod"],
        default="reve",
        help="Type of preprocessing to apply",
    )
    args = parser.parse_args()

    rootDir = args.root
    processedDir = args.processed

    files_H, files_MDD = iter_files(rootDir)
    files_H = sorted(files_H)
    files_MDD = sorted(files_MDD)
    print(files_H)
    print(files_MDD)
    print(len(files_H), len(files_MDD))

    files_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    dataset = {
        "train": [],
        "val": [],
        "test": [],
    }

    files_dict["train"].extend(files_H[:40])
    files_dict["train"].extend(files_MDD[:42])
    files_dict["val"].extend(files_H[40:48])
    files_dict["val"].extend(files_MDD[42:52])
    files_dict["test"].extend(files_H[48:])
    files_dict["test"].extend(files_MDD[52:])

    print(files_dict["train"])
    print(files_dict["val"])
    print(files_dict["test"])

    os.makedirs(processedDir, exist_ok=True)
    db = lmdb.open(processedDir, map_size=1273741824)

    for files_key, files_list in files_dict.items():
        for file in files_list:
            raw = mne.io.read_raw_edf(os.path.join(rootDir, file), preload=True)
            print(raw.info["ch_names"])
            raw.pick_channels(selected_channels, ordered=True)
            print(raw.info["ch_names"])
            raw.resample(200)
            raw.filter(l_freq=0.3, h_freq=30)
            raw.notch_filter((50))
            # raw.plot_psd(average=True)
            eeg_array = raw.to_data_frame().values
            # print(raw.info)
            eeg_array = eeg_array[:, 1:]
            points, chs = eeg_array.shape
            print(eeg_array.shape)
            a = points % (5 * 200)
            print(a)
            if a != 0:
                eeg_array = eeg_array[:-a, :]
            eeg_array = eeg_array.reshape(-1, 5, 200, chs)
            eeg_array = eeg_array.transpose(0, 3, 1, 2)
            print(eeg_array.shape)

            if args.type == "reve":
                eeg_array = rearrange(eeg_array, "b c t d -> b c (t d)", d=200)
            # else do nothing, as it is already in the desired format

            label = 1 if "MDD" in file else 0
            for i, sample in enumerate(eeg_array):
                sample_key = f"{file[:-4]}_{i}"
                data_dict = {"sample": sample, "label": label}
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()
