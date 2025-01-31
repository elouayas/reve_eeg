"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
import os
import pickle

import lmdb
import mne


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet dataset")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the PhysioNet dataset",
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

    tasks = ["04", "06", "08", "10", "12", "14"]  # select the data for motor imagery

    root_dir = args.root
    files = list(os.listdir(root_dir))
    files = [file for file in files if file.startswith("S") and len(file) == 4]
    files = sorted(files)

    files_dict = {
        "train": files[:70],
        "val": files[70:89],
        "test": files[89:109],
    }

    print(files_dict)
    dataset = {
        "train": [],
        "val": [],
        "test": [],
    }

    selected_channels = [
        "Fc5.",
        "Fc3.",
        "Fc1.",
        "Fcz.",
        "Fc2.",
        "Fc4.",
        "Fc6.",
        "C5..",
        "C3..",
        "C1..",
        "Cz..",
        "C2..",
        "C4..",
        "C6..",
        "Cp5.",
        "Cp3.",
        "Cp1.",
        "Cpz.",
        "Cp2.",
        "Cp4.",
        "Cp6.",
        "Fp1.",
        "Fpz.",
        "Fp2.",
        "Af7.",
        "Af3.",
        "Afz.",
        "Af4.",
        "Af8.",
        "F7..",
        "F5..",
        "F3..",
        "F1..",
        "Fz..",
        "F2..",
        "F4..",
        "F6..",
        "F8..",
        "Ft7.",
        "Ft8.",
        "T7..",
        "T8..",
        "T9..",
        "T10.",
        "Tp7.",
        "Tp8.",
        "P7..",
        "P5..",
        "P3..",
        "P1..",
        "Pz..",
        "P2..",
        "P4..",
        "P6..",
        "P8..",
        "Po7.",
        "Po3.",
        "Poz.",
        "Po4.",
        "Po8.",
        "O1..",
        "Oz..",
        "O2..",
        "Iz..",
    ]

    os.makedirs(args.processed, exist_ok=True)
    db = lmdb.open(args.processed, map_size=4614542346)

    for files_key, files_list in files_dict.items():
        for file in files_list:
            for task in tasks:
                raw_path = os.path.join(root_dir, file, f"{file}R{task}.edf")
                if not os.path.exists(raw_path):
                    print(f"File {raw_path} does not exist, skipping...")
                    continue
                raw = mne.io.read_raw_edf(raw_path, preload=True)
                raw.pick_channels(selected_channels, ordered=True)
                if len(raw.info["bads"]) > 0:
                    print("interpolate_bads")
                    raw.interpolate_bads()
                raw.set_eeg_reference(ref_channels="average")
                raw.filter(l_freq=0.3, h_freq=None)
                raw.notch_filter((60))
                raw.resample(200)
                events_from_annot, event_dict = mne.events_from_annotations(raw)
                epochs = mne.Epochs(
                    raw,
                    events_from_annot,
                    event_dict,
                    tmin=0,
                    tmax=4.0 - 1.0 / raw.info["sfreq"],
                    baseline=None,
                    preload=True,
                )
                data = epochs.get_data(units="uV")
                events = epochs.events[:, 2]
                print(data.shape, events)
                data = data[:, :, -800:]
                bz, ch_nums, _ = data.shape
                if args.type == "cbramod":
                    data = data.reshape(bz, ch_nums, 4, 200)
                # else do nothing
                print(data.shape)
                for i, (sample, event) in enumerate(zip(data, events)):
                    if event != 1:
                        sample_key = f"{file}R{task}-{i}"
                        data_dict = {"sample": sample, "label": event - 2 if task in ["04", "08", "12"] else event}
                        txn = db.begin(write=True)
                        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                        txn.commit()
                        dataset[files_key].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
    txn.commit()
    db.close()
