"""
adapted from https://github.com/935963004/NeuroLM
"""

import argparse
import os
import pickle

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=str, required=True, help="Raw data directory")
parser.add_argument("--processed", type=str, required=True, help="Processed data directory")
args = parser.parse_args()


N_JOBS = min(16, os.cpu_count() - 1)

drop_channels = ["EMG chin", "EOG E1-M2", "EOG E2-M2", "ECG"]
chOrder_standard = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]

symbols_hmc = {
    " Sleep stage W": 0,
    " Sleep stage N1": 1,
    " Sleep stage N2": 2,
    " Sleep stage N3": 3,
    " Sleep stage R": 4,
}


def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape
    fs = 200.0
    [numChan, numPoints] = signals.shape

    features = np.zeros([numEvents - 2, numChan, int(fs) * 30])
    labels = np.zeros([numEvents - 2, 1])
    i = 0
    for _, row in EventData.iterrows():
        if row[" Duration"] != 30:
            continue
        start = np.where((times) >= row[" Recording onset"])[0][0]
        end = np.where((times) >= (row[" Recording onset"] + row[" Duration"]))[0][0]
        features[i, :] = signals[:, start:end]
        labels[i, :] = symbols_hmc[row[" Annotation"]]
        i += 1
    return [features, labels]


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0, n_jobs=N_JOBS)
    Rawdata.notch_filter(50.0, n_jobs=N_JOBS)
    Rawdata.resample(200, n_jobs=N_JOBS)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units="uV")
    labelFile = fileName[0:-4] + "_sleepscoring.txt"
    labelData = pd.read_csv(labelFile)
    Rawdata.close()
    return [signals, times, labelData, Rawdata]


def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("\t%s" % fname)
        try:
            [signals, times, labelData, Rawdata] = readEDF(fname)
        except (ValueError, KeyError):
            print("something funky happened in " + fname)
            continue
        signals, labels = BuildEvents(signals, times, labelData)

        for idx, (signal, label) in tqdm(enumerate(zip(signals, labels))):
            sample = {
                "X": signal,
                "ch_names": [name.split(" ")[-1].split("-")[0] for name in chOrder_standard],
                "y": label,
            }
            print(signal.shape)

            save_pickle(
                sample,
                os.path.join(OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"),
            )

    return Features, Labels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


# root = "/D_data/weibangjiang/HMC/physionet.org/files/hmc-sleep-staging/1.1/recordings"
# out_dir = "../teamdrive/HMC"

root = args.raw
out_dir = args.processed

train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "eval")
test_out_dir = os.path.join(out_dir, "test")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(test_out_dir):
    os.makedirs(test_out_dir)

edf_files = []
for dirName, subdirList, fileList in os.walk(root):
    for fname in fileList:
        if len(fname) == 9 and fname[-4:] == ".edf":
            edf_files.append(os.path.join(dirName, fname))
edf_files.sort()

train_files = edf_files[:100]
eval_files = edf_files[100:125]
test_files = edf_files[125:]

fs = 200
TrainFeatures = np.empty((0, 4, fs * 30))  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
load_up_objects(train_files, TrainFeatures, TrainLabels, train_out_dir)

fs = 200
EvalFeatures = np.empty((0, 4, fs * 30))  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
load_up_objects(eval_files, EvalFeatures, EvalLabels, eval_out_dir)

fs = 200
TestFeatures = np.empty((0, 4, fs * 30))  # 0 for lack of intialization, 22 for channels, fs for num of points
TestLabels = np.empty([0, 1])
load_up_objects(test_files, TestFeatures, TestLabels, test_out_dir)
