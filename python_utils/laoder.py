from torch.utils.data import Dataset, DataLoader
from sklearn import decomposition
import numpy as np
from data_calibration import calibrate_amplitude, dwn_noise, hampel

SUBCARRIES_NUM = 52
PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582

def read_csi_data(amplitude):
    subcarries_num = SUBCARRIES_NUM
    amplitudes_list=[]
    # print('amplitude',amplitude)
    for data in amplitude:
        print('data',data)
        if len(data) != subcarries_num :
            raise ValueError(f"Data length mismatch: expected {subcarries_num}, got {len(data)}")

        amplitudes = data[:subcarries_num]
        amplitudes_list.append(amplitudes)
    return np.array(amplitudes_list)

class CSIDataset(Dataset):
    """CSI Dataset for inference without labels."""

    def __init__(self, amplitudes, window_size=32, step=1, is_training=False):
        self.is_training = is_training
        # print('amplitudes',amplitudes)
        self.amplitudes=read_csi_data(amplitudes)
        self.amplitudes = calibrate_amplitude(self.amplitudes)
        pca = decomposition.PCA(n_components=12)
        self.amplitudes_pca = []

        data_len = self.amplitudes.shape[0]
        print('Data length:', data_len)
        print('Amplitudes shape:', self.amplitudes.shape)   
        for i in range(self.amplitudes.shape[1]):
            self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[:data_len]

        self.amplitudes_pca = pca.fit_transform(self.amplitudes)
        self.amplitudes_pca = np.array(self.amplitudes_pca)

        self.window = window_size
        self.step = step

    def __getitem__(self, idx):
        idx = idx * self.step
        all_xs = []

        for index in range(idx, idx + self.window):
            amplitude = self.amplitudes[index]
            pca = self.amplitudes_pca[index]
            combined = np.append(amplitude, pca)

            if self.is_training:
                noise = np.random.normal(0, 0.01, size=combined.shape)
                combined += noise

            all_xs.append(combined)

        return np.array(all_xs)

    def __len__(self):
        length = max(1, int((self.amplitudes.shape[0] - self.window) // self.step) + 1)
        print(f"CSIDataset length: {length}")
        return length

