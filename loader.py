import h5py
import numpy as np
from torch.utils.data import Dataset


class FeatureLoader(Dataset):

    def __init__(self, data_set):
        self.data_set=data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):

        sample = self.data_set[idx]

        with h5py.File(sample["X"], "r") as f:
            X = f['data'][()]

        Y = np.expand_dims(sample["Y"], 0)

        return {"X": X.astype(np.float32),
                "Y": Y.astype(np.float32)}
