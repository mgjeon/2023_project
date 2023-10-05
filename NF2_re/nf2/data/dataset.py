import numpy as np
import torch
from torch.utils.data import Dataset


class BatchesDataset(Dataset):
    def __init__(self, batches_file_paths, batch_size):
        self.batches_file_paths = batches_file_paths
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(
            np.load(list(self.batches_file_paths.values())[0], mmap_mode="r").shape[0]
            / self.batch_size
        ).astype(np.int32)

    def __getitem__(self, idx):
        data = {
            k: np.copy(
                np.load(bf, mmap_mode="r")[
                    idx * self.batch_size : (idx + 1) * self.batch_size
                ]
            )
            for k, bf in self.batches_file_paths.items()
        }
        return data


class RandomCoordinateDataset(Dataset):
    def __init__(self, cube_shape, spatial_norm, batch_size):
        super().__init__()
        cube_shape = np.array(
            [[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]]
        )
        self.cube_shape = cube_shape
        self.spatial_norm = spatial_norm
        self.batch_size = batch_size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        random_coords = torch.FloatTensor(self.batch_size, 3).uniform_()
        for i in range(3):
            random_coords[:, i] = (
                random_coords[:, i] * (self.cube_shape[i, 1] - self.cube_shape[i, 0])
                + self.cube_shape[i, 0]
            )
        random_coords = random_coords / self.spatial_norm
        return random_coords


class CubeDataset(Dataset):
    def __init__(self, cube_shape, spatial_norm, batch_size=1024, strides=1):
        coords = np.stack(
            np.mgrid[
                : cube_shape[0] : strides,
                : cube_shape[1] : strides,
                : cube_shape[2] : strides,
            ],
            -1,
        )
        self.coords_shape = coords.shape[:-1]
        coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
        coords = coords.reshape((-1, 3))
        self.coords = np.split(coords, np.arange(batch_size, len(coords), batch_size))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx]
