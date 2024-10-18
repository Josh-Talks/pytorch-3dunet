import collections

collections.Sequence = collections.abc.Sequence
import os

import imageio
import numpy as np
import torch

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats, read_file_names
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("DSB2018Dataset")


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DSB2018Dataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, expand_dims=True):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "images")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir

        stats = calculate_stats(self.images, True)

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(root_dir, "masks")
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, expand_dims)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != "test":
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        file_paths = phase_config["file_paths"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [cls(file_paths[0], phase, transformer_config, expand_dims)]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths


class HoechstDataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, expand_dims=True):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "images/png")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims, rgb=False)
        self.file_path = images_dir

        stats = calculate_stats(self.images, True)

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(root_dir, "annotations")
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, expand_dims, rgb=True)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != "test":
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        file_paths = phase_config["file_paths"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [cls(file_paths[0], phase, transformer_config, expand_dims)]

    @staticmethod
    def _load_files(dir, expand_dims, rgb):
        files_data = []
        paths = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if rgb:
                img = transforms.RgbToLabel()(img)
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths


class BBBC039Dataset(ConfigDataset):
    def __init__(self, file_names_path, phase, transformer_config, expand_dims=True):
        base_dir = os.path.dirname(file_names_path)
        assert os.path.isdir(base_dir), f"{base_dir} is not a directory"
        assert phase in ["train", "val", "test"]
        assert phase == os.path.basename(file_names_path).split(".")[0]

        self.phase = phase

        self.file_names = read_file_names(file_names_path)
        # load raw images
        images_dir = os.path.join(base_dir, "images")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(
            images_dir, self.file_names, expand_dims, "tif"
        )
        self.file_path = images_dir

        stats = calculate_stats(self.images, True)

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(base_dir, "masks")
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(
                masks_dir, self.file_names, expand_dims, "png"
            )
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != "test":
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        file_paths = phase_config["file_paths"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [cls(file_paths[0], phase, transformer_config, expand_dims)]

    @staticmethod
    def _load_files(dir, file_names, expand_dims, file_type):
        files_data = []
        paths = []
        for file in file_names:
            path = os.path.join(dir, file + "." + file_type)
            img = np.asarray(imageio.imread(path))
            if img.ndim == 3:
                img = img[:, :, 0]
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths
