"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
import os
import pickle

import lmdb
import numpy as np
from einops import rearrange  # noqa
from scipy import signal


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="Root directory")
parser.add_argument("--processed", type=str, required=True, help="Processed data directory")
args = parser.parse_args()

labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
root_dir = args.root
files = list(os.listdir(root_dir))
files = sorted(files)

files_dict = {
    "train": files[:80],
    "val": files[80:100],
    "test": files[100:],
}

dataset = {
    "train": [],
    "val": [],
    "test": [],
}

path = args.processed.replace("processed", "processed_cbramod")

os.makedirs(path, exist_ok=True)
db = lmdb.open(path, map_size=6612500172)

for files_key, files_list in files_dict.items():
    for file in files_list:
        with open(os.path.join(root_dir, file), "rb") as f:
            array = pickle.load(f)
        eeg = signal.resample(array, 6000, axis=2)
        # eeg = rearrange(eeg, "b c (t d) -> b c t d", d=2000)
        eeg_ = eeg.reshape(28, 32, 30, 200)
        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10 * j : 10 * (j + 1), :]
                # sample = samples[:, j : (j + 1), :].squeeze(1)
                sample_key = f"{file}-{i}-{j}"
                print(sample_key)
                data_dict = {"sample": sample, "label": label}
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
