import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from nnf2.data.dataset import BatchesDataset, CubeDataset, RandomCoordinateDataset
from nnf2.data.isee import nlfff
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler


class SlicesDataModule(LightningDataModule):
    def __init__(
        self,
        b_slices,
        height,
        spatial_norm,
        b_norm,
        work_directory,
        batch_size={"boundary": 1e4, "random": 2e4},
        iterations=1e5,
        num_workers=None,
        error_slices=None,
        height_mapping={"z": [0]},
        boundary={"type": "open"},
        validation_strides=1,
        meta_data=None,
        plot_overview=True,
        Mm_per_pixel=None,
        buffer=None,
        **kwargs
    ):
        super().__init__()

        # plot
        if plot_overview:
            for i in range(b_slices.shape[2]):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                for j in range(3):
                    axs[j].imshow(b_slices[..., i, j].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')

        # data parameters
        self.height = height
        self.spatial_norm = spatial_norm
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        self.Mm_per_pixel = Mm_per_pixel

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        # load dataset
        os.makedirs(work_directory, exist_ok=True)

        coords = np.stack(
            np.mgrid[: b_slices.shape[0], : b_slices.shape[1], : b_slices.shape[2]], -1
        ).astype(np.float32)

        cube_shape = [*b_slices.shape[:2], height]
        self.cube_shape = cube_shape

        # flatten data
        coords = coords.reshape((-1, 3)).astype(np.float32)
        values = b_slices.reshape((-1, 3)).astype(np.float32)

        # normalize data
        coords = coords / spatial_norm
        values = values / b_norm

        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, "coords.npy")
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, "values.npy")
        np.save(values_npy_path, values)

        batches_file_paths = {"coords": coords_npy_path, "values": values_npy_path}

        boundary_batch_size = (
            int(batch_size["boundary"])
            if isinstance(batch_size, dict)
            else int(batch_size)
        )
        random_batch_size = (
            int(batch_size["random"])
            if isinstance(batch_size, dict)
            else int(batch_size)
        )

        # create data loader
        self.dataset = BatchesDataset(batches_file_paths, boundary_batch_size)
        self.random_dataset = RandomCoordinateDataset(
            cube_shape, spatial_norm, random_batch_size
        )
        self.cube_dataset = CubeDataset(
            cube_shape,
            spatial_norm,
            batch_size=boundary_batch_size,
            strides=validation_strides,
        )

    def train_dataloader(self):
        data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomSampler(
                self.dataset, replacement=True, num_samples=self.iterations
            ),
        )

        random_loader = DataLoader(
            self.random_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomSampler(
                self.random_dataset, replacement=True, num_samples=self.iterations
            ),
        )

        return {"boundary": data_loader, "random": random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(
            self.cube_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        boundary_loader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return [cube_loader, boundary_loader]


class ISEEDataModule(SlicesDataModule):
    def __init__(
        self,
        data_path,
        height=None,
        slices=None,
        bin=1,
        use_bz=False,
        components=False,
        *args,
        **kwargs
    ):
        data = nlfff(data_path)
        b_slices = np.stack([data.bx, data.by, data.bz], -1)
        if height is None:
            height = b_slices.shape[2]
        else:
            height = height

        if slices:
            b_slices = b_slices[:, :, slices]

        
        Mm_per_pixel = np.array(
            [np.diff(data.x)[0], np.diff(data.y)[0], np.diff(data.z)[0]]
        )

        super().__init__(b_slices, height, Mm_per_pixel=Mm_per_pixel, *args, **kwargs)
