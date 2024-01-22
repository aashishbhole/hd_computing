import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import welch
from scipy.stats import skew, kurtosis, chi2

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
import copy

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
# from eeg import EEG
import pyeeg

import pyedflib

device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
SAMP_FREQ = 256
WINDOW = 2
T0 = 0
T1 = 1000
T2 = 772
T3 = 3000


class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


def extract_spectral_features(eeg_time_series):
    mean = np.mean(eeg_time_series)
    standard_dev = np.std(eeg_time_series)
    Kmax = 5
    Tau = 4
    DE = 10
    M = 10
    R = 0.3 * standard_dev
    Band = np.arange(1, 86, 2)
    Fs = 256
    # DFA = pyeeg.dfa(eeg_time_series)
    HFD = pyeeg.hfd(eeg_time_series, Kmax)
    SVD_Entropy = pyeeg.svd_entropy(eeg_time_series, Tau, DE)
    Fisher_Information = pyeeg.fisher_info(eeg_time_series, Tau, DE)
    # ApEn = pyeeg.ap_entropy(eeg_time_series, M, R)
    p, p_ratio = pyeeg.bin_power(eeg_time_series, Band, Fs)
    Spectral_Entropy = pyeeg.spectral_entropy(eeg_time_series, Band, Fs, Power_Ratio=p_ratio)
    PFD = pyeeg.pfd(eeg_time_series)

    return np.array([HFD, SVD_Entropy, Fisher_Information, Spectral_Entropy, PFD])


def extract_features(eeg_signal, fs=256, epoch_duration=2, num_epochs=1800):
    samples_per_epoch = fs * epoch_duration

    # Calculate the number of samples for all epochs
    total_samples = num_epochs * samples_per_epoch

    # Reshape the EEG signal to have one epoch per column
    num_channels = eeg_signal.shape[0]
    eeg_epochs = eeg_signal[:, :total_samples].reshape(num_channels, num_epochs, samples_per_epoch)
    eeg_epochs = eeg_epochs.transpose((1, 0, 2))

    features = np.array([np.mean([extract_spectral_features(channel) for channel in epoch], axis=0)
                         for epoch in eeg_epochs])

    return features


def read_eeg_data(filename):
    f = pyedflib.EdfReader(filename)
    num_signals = f.signals_in_file
    signals = [f.readSignal(i) for i in range(num_signals)]
    f.close()
    return np.array(signals)


# Function to compute Mahalanobis distance
def mahalanobis_distance(x, mean, covariance):
    x_minus_mean = x - mean
    inv_covariance = np.linalg.inv(covariance)
    distance = np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance), x_minus_mean.T))
    return distance


data_1 = read_eeg_data('./data/eeg/chb_mit/chb01/chb01_15.edf')
data_2 = read_eeg_data('./data/eeg/chb_mit/chb01/chb01_16.edf')
data = np.concatenate((data_1, data_2), axis=1)
features_T1 = extract_features(data[:, T0 * SAMP_FREQ:(T0 + T1) * SAMP_FREQ], fs=SAMP_FREQ, epoch_duration=WINDOW,
                               num_epochs=(T1 // WINDOW))
features_T2 = extract_features(data[:, (T0 + T1) * SAMP_FREQ:(T0 + T1 + T2) * SAMP_FREQ], fs=SAMP_FREQ,
                               epoch_duration=WINDOW,
                               num_epochs=(T2 // WINDOW))
features_T3 = extract_features(data[:, (T0 + T1 + T2) * SAMP_FREQ:(T0 + T1 + T2 + T3) * SAMP_FREQ], fs=SAMP_FREQ,
                               epoch_duration=WINDOW,
                               num_epochs=(T3 // WINDOW))
print(features_T1.shape)
print(features_T2.shape)
print(features_T3.shape)

# Gaussian Estimation for T1 Phase
num_components = 2
gmm = GaussianMixture(n_components=num_components, random_state=0)
gmm.fit(features_T1)

# Get the means and covariance matrices for each component
means = gmm.means_
covariances = gmm.covariances_
confidence_level = 0.95
threshold = chi2.ppf(confidence_level, df=num_components)
print("GMM Means:", means)
print("Threshold:", threshold)

# GMM evaluation and HD Model training for T2 Phase
features_T2 = torch.from_numpy(features_T2)

encode = Encoder(DIMENSIONS, features_T2.shape[-1])
encode = encode.to(device)

num_classes = 2
model = Centroid(DIMENSIONS, num_classes)  # hd_model = HDClustering(num_clusters)
model = model.to(device)  # hd_model = hd_model.to(device)

outliers = 0
with torch.no_grad():
    for idx, sample in tqdm(enumerate(features_T2), desc="Training"):
        distances = [mahalanobis_distance(sample, means[j], covariances[j]) for j in range(num_components)]
        min_distance = min(distances)

        sample = sample.to(device)
        sample_hv = encode(sample).unsqueeze(0)
        # Compare the minimum Mahalanobis distance to the threshold to check for outlier
        if min_distance > threshold:
            outliers += 1
            print("outlier at", idx)
            model.add(sample_hv, torch.tensor([1]).to(device))
        else:
            model.add(sample_hv, torch.tensor([0]).to(device))
print("Num of outliers:", outliers)
print("Model Weights Trained:", model.weight)

# Gaussian/HD Model evaluation for T3 Phase
features_T3 = torch.from_numpy(features_T3)
labels = torch.zeros(features_T3.size(0), 1)
labels[1422:1447] = 1

accuracy = torchmetrics.Accuracy("binary", num_classes=num_classes)
precision = torchmetrics.Precision("binary", num_classes=num_classes)
recall = torchmetrics.Recall("binary", num_classes=num_classes)
f1_score = torchmetrics.F1Score("binary", num_classes=num_classes)
specificity = torchmetrics.Specificity("binary", num_classes=num_classes)

preds = []
with torch.no_grad():
    for sample, label in tqdm(zip(features_T3, labels), desc="Testing"):
        # distances = [mahalanobis_distance(sample, means[j], covariances[j]) for j in range(num_components)]
        # min_distance = min(distances)
        # if min_distance > np.sqrt(threshold):
        #     gmm_pred = torch.tensor(1)
        # else:
        #     gmm_pred = torch.tensor(0)
        sample = sample.to(device)
        sample_hv = encode(sample).unsqueeze(0)
        pred = model(sample_hv).cpu().argmax(1)
        preds.append(pred[0])
        accuracy.update(pred, label)
        precision.update(pred, label)
        recall.update(pred, label)
        f1_score.update(pred, label)
        specificity.update(pred, label)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print(f"Testing precision of {(precision.compute().item() * 100):.3f}%")
print(f"Testing recall/sensitivity of {(recall.compute().item() * 100):.3f}%")
print(f"Testing F1 score of {(f1_score.compute().item() * 100):.3f}%")
print(f"Testing specificity of {(specificity.compute().item() * 100):.3f}%")

print("Predicted Loc:", torch.nonzero(torch.tensor(preds), as_tuple=False))
print(len(preds))
