import os
import os.path
from typing import Callable, Optional, Tuple, List
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import pyeeg
from torchhd.datasets.utils import download_file, unzip_file


class EEG(data.Dataset):
    """`EEG dataset.

    Args:
        name (string): Name of the dataset
        root (string): Root directory of dataset where ``train.data``
            and  ``test.data`` exist.
        train (bool, optional): If True, creates dataset from ``train.data``,
            otherwise from ``test.data``.
        transform (callable, optional): A function/transform that takes in an torch.FloatTensor
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    classes: List[str] = [
        "NS",
        "S",
    ]

    def __init__(
            self,
            name: str,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        root = os.path.join(root, "eeg")
        root = os.path.join(root, name)
        root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Download it to the root directory specified."
            )

        self._load_data()

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: (sample, target) where target is the index of the target class
        """
        sample = self.data[index]
        label = self.targets[index]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label

    def _check_integrity(self) -> bool:
        if not os.path.isdir(self.root):
            return False

        # Check if the root directory contains the required files
        has_train_file = os.path.isfile(os.path.join(self.root, "train.data"))
        has_test_file = os.path.isfile(os.path.join(self.root, "test.data"))
        if has_train_file and has_test_file:
            return True

        # TODO: Add more specific checks like an MD5 checksum

        return False

    def _load_data(self):
        data_file = "train.data" if self.train else "test.data"
        data = pd.read_csv(os.path.join(self.root, data_file), header=None)
        self.data, self.targets = self.process_eeg_data(data)

    @staticmethod
    def extract_features(eeg_time_series):
        mean = np.mean(eeg_time_series)
        standard_dev = np.std(eeg_time_series)
        Kmax = 5
        Tau = 4
        DE = 10
        M = 10
        R = 0.3 * standard_dev
        Band = np.arange(1, 86, 2)
        Fs = 173
        # DFA = pyeeg.dfa(eeg_time_series)
        HFD = pyeeg.hfd(eeg_time_series, Kmax)
        SVD_Entropy = pyeeg.svd_entropy(eeg_time_series, Tau, DE)
        Fisher_Information = pyeeg.fisher_info(eeg_time_series, Tau, DE)
        # ApEn = pyeeg.ap_entropy(eeg_time_series, M, R)
        p, p_ratio = pyeeg.bin_power(eeg_time_series, Band, Fs)
        Spectral_Entropy = pyeeg.spectral_entropy(eeg_time_series, Band, Fs, Power_Ratio=p_ratio)
        PFD = pyeeg.pfd(eeg_time_series)

        return np.array([HFD, SVD_Entropy, Fisher_Information, Spectral_Entropy, PFD, mean])

    def process_eeg_data(self, dataframe):
        X = dataframe.values[:, :-1]
        y = dataframe.values[:, -1]
        features = []
        for i in range(len(X)):
            features.append(self.extract_features(X[i]))
        return torch.tensor(np.array(features), dtype=torch.float), torch.tensor(y, dtype=torch.long)
