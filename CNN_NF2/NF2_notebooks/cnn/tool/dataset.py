from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, batches_file_paths, batch_size):
        self.batches_file_paths = batches_file_paths
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(np.load(list(self.batches_file_paths.values())[0], mmap_mode='r').shape[0] / self.batch_size).astype(np.int32)

    def __getitem__(self, idx):
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data