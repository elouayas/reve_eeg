"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
import os
import pickle

import lmdb
import mne
import numpy as np
from scipy.signal import butter, lfilter, resample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="Raw data directory")
    parser.add_argument("--processed", type=str, required=True, help="Processed data directory")
    parser.add_argument(
        "--type",
        type=str,
        choices=["reve", "cbramod"],
        default="reve",
        help="Type of preprocessing to apply",
    )
    args = parser.parse_args()

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype="band")

    files_dict = {
        "train": [
            "A01E.gdf",
            "A01T.gdf",
            "A02E.gdf",
            "A02T.gdf",
            "A03E.gdf",
            "A03T.gdf",
            "A04E.gdf",
            "A04T.gdf",
            "A05E.gdf",
            "A05T.gdf",
        ],
        "val": ["A06E.gdf", "A06T.gdf", "A07E.gdf", "A07T.gdf"],
        "test": ["A08E.gdf", "A08T.gdf", "A09E.gdf", "A09T.gdf"],
    }

    classes = {
        769: 0,  # left hand
        770: 1,  # right hand
        771: 2,  # feet
        772: 3,  # tongue
    }

    dataset = {"train": [], "val": [], "test": []}
    root_dir = args.raw
    os.makedirs(args.processed, exist_ok=True)
    db = lmdb.open(args.processed, map_size=1024**3)  # 1 GB

    for split, files_list in files_dict.items():
        for file in files_list:
            if not os.path.exists(os.path.join(root_dir, file)):
                print(f"File {file} not found in {root_dir}. Skipping.")
                continue

            print(f"Processing {file}...")
            raw = mne.io.read_raw_gdf(os.path.join(root_dir, file), preload=True)
            raw.pick_types(eeg=True)

            raw_data = raw.get_data(units="uV")

            events, event_id = mne.events_from_annotations(raw)
            print(f"Events found: {events.shape[0]}")

            cue_events = []
            cue_labels = []

            for e in events:
                event_type_int = e[2]
                desc = None
                for k, v in event_id.items():
                    if v == event_type_int:
                        desc = k
                        break

                if desc is not None and desc.isdigit() and int(desc) in classes:
                    cue_events.append(e[0])
                    cue_labels.append(classes[int(desc)])

            for i in range(len(cue_events)):
                start_idx = cue_events[i]
                end_idx = cue_events[i + 1] if i < len(cue_events) - 1 else raw_data.shape[-1]

                sample = raw_data[:22, start_idx:end_idx]  # truncate to 22 channels

                sample = sample - np.mean(sample, axis=0, keepdims=True)

                b, a = butter_bandpass(0.3, 50, 250)
                sample = lfilter(b, a, sample, axis=-1)

                sample = sample[:, 2 * 250 : 6 * 250]
                sample = resample(sample, 800, axis=-1)

                if args.type == "cbramod":
                    sample = sample.reshape(22, 4, 200)

                label = cue_labels[i]
                sample_key = f"{file[:-4]}-{i}"
                data_dict = {
                    "sample": sample,
                    "label": label,
                }

                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[split].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()
