import collections

collections.Sequence = collections.abc.Sequence
import os
import glob
from abc import abstractmethod

import imageio.v2 as imageio
import numpy as np
import torch
import h5py

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.hdf5 import _create_padded_indexes
from pytorch3dunet.datasets.utils import (
    ConfigDataset,
    calculate_stats,
    read_file_names,
    get_roi_slice,
    get_slice_builder,
    mirror_pad,
)
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("DSB2018Dataset")


def traverse_S_BIAD1410_paths(file_paths):
    """
    Traverse the given list of file paths and include all non mask tif files found in the directories.
    """
    assert isinstance(file_paths, list), "file_paths should be a list of strings"
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # find all files in directory with ending .tif and not containing "mask"
            paths = glob.glob(os.path.join(file_path, "**/*.tif"), recursive=True)
            condition = lambda x: "mask" not in os.path.basename(x)
            paths = list(filter(condition, paths))
            results.extend(paths)
        else:
            results.append(file_path)
    return results


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
    def __init__(
        self,
        root_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "images")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

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
        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
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
    def __init__(
        self,
        root_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "images/png")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims, rgb=False)
        self.file_path = images_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

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
        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

    @staticmethod
    def _load_files(dir, expand_dims, rgb):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
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
    def __init__(
        self,
        file_names_path,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
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

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

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

        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

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


class S_BIAD634_Dataset(ConfigDataset):
    def __init__(
        self,
        file_names_path,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_dir="rawimages",
        label_dir="groundtruth",
    ):
        base_dir = os.path.dirname(file_names_path)
        assert os.path.isdir(base_dir), f"{base_dir} is not a directory"
        assert phase in ["train", "val", "test"]
        assert phase == os.path.basename(file_names_path).split(".")[0]

        self.phase = phase

        self.file_names = read_file_names(file_names_path)
        # load raw images
        images_dir = os.path.join(base_dir, image_dir)
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(
            images_dir, self.file_names, expand_dims, "tif"
        )
        self.file_path = images_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(base_dir, label_dir)
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(
                masks_dir, self.file_names, expand_dims, "tif"
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

        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
                dataset_config.get("image_dir", "rawimages"),
                dataset_config.get("label_dir", "groundtruth"),
            )
        ]

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


class S_BIAD895_Dataset(ConfigDataset):
    def __init__(
        self,
        root_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "Raw")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(root_dir, "Masks")
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
        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
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


class S_BIAD1410_Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to tif file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
        self,
        file_path,
        roi,
        phase,
        slice_builder_config,
        transformer_config,
        label_suffix="mask",
        global_normalization=True,
        global_percentiles=None,
    ):
        assert phase in ["train", "val", "test"]

        self.phase = phase
        self.file_path = file_path
        self.label_file_path = file_path.replace(".tif", f"_{label_suffix}.tif")
        if roi is not None:
            self.roi = get_roi_slice(roi)
        else:
            self.roi = roi
        self.patch_shape = slice_builder_config.get("patch_shape")
        self.halo_shape = slice_builder_config.get("halo_shape", [0, 0, 0])

        if global_normalization:
            logger.info("Calculating mean and std of the raw data...")
            self.raw = imageio.volread(file_path)
            if self.roi is not None:
                self.raw = self.raw[self.roi]
            if global_percentiles is not None:
                stats = calculate_stats(
                    self.raw,
                    percentile_min=global_percentiles[0],
                    percentile_max=global_percentiles[1],
                )
            else:
                stats = calculate_stats(self.raw)
        else:
            self.raw = None
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != "test":
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

            self._check_volume_sizes()
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None

            # compare patch and stride configuration
            patch_shape = slice_builder_config.get("patch_shape")
            stride_shape = slice_builder_config.get("stride_shape")
            if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                logger.warning(
                    f"Found non-zero halo shape {self.halo_shape}. "
                    f"In this case: patch shape and stride shape should be equal for optimal prediction "
                    f"performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!"
                )

        if self.roi is not None:
            if self.raw is None:
                self.raw = imageio.volread(file_path)[self.roi]
            self.label = (
                imageio.volread(self.label_file_path)[self.roi]
                if phase != "test"
                else None
            )
            weight_map = None

        else:
            if self.raw is None:
                self.raw = imageio.volread(file_path)
            self.label = (
                imageio.volread(self.label_file_path) if phase != "test" else None
            )
            weight_map = None
        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(
            self.raw, self.label, weight_map, slice_builder_config
        )
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f"Number of patches: {self.patch_count}")

    @abstractmethod
    def get_raw_patch(self, idx):
        return self.raw[idx]

    @abstractmethod
    def get_label_patch(self, idx):
        return self.label[idx]

    @abstractmethod
    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            self._raw_padded = mirror_pad(self.raw, self.halo_shape)
        return self._raw_padded[idx]

    def volume_shape(self):
        raw = imageio.volread(self.file_path)
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == "test":
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[
                    1:
                ]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(
                    raw_idx, self.halo_shape
                )
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            raw_patch_transformed = self.raw_transform(
                self.get_raw_padded_patch(raw_idx_padded)
            )
            return raw_patch_transformed, raw_idx
        else:
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(
                self.get_label_patch(label_idx)
            )
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    def _check_volume_sizes(self):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        raw = imageio.volread(self.file_path)
        label = imageio.volread(self.label_file_path)
        assert raw.ndim in [3, 4], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
        assert label.ndim in [3, 4], "Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
        assert _volume_shape(raw) == _volume_shape(
            label
        ), "Raw and labels have to be of the same size"

    def get_patch_shape(self):
        return self.patch_shape

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load slice builder config
        slice_builder_config = phase_config["slice_builder"]
        # load files to process
        file_paths = phase_config["file_paths"]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_S_BIAD1410_paths(file_paths)
        roi = phase_config.get("roi", None)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f"Loading {phase} set from: {file_path}...")
                dataset = cls(
                    file_path=file_path,
                    roi=roi,
                    phase=phase,
                    slice_builder_config=slice_builder_config,
                    transformer_config=transformer_config,
                    label_suffix=phase_config.get("label_suffix", "mask"),
                    global_normalization=dataset_config.get(
                        "global_normalization", None
                    ),
                    global_percentiles=dataset_config.get("global_percentiles", None),
                )
                datasets.append(dataset)
            except Exception:
                logger.error(f"Skipping {phase} set: {file_path}", exc_info=True)
        return datasets

    

### General Datasets

class Abstract_TIF_Dataset(ConfigDataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        filenames_path=None,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        assert os.path.isdir(image_dir), f"{image_dir} is not a directory"
        assert os.path.isdir(mask_dir), f"{mask_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        assert os.path.isdir(image_dir)

        if filenames_path is not None:
            self.file_names = read_file_names(filenames_path)

        self.images, self.paths = self._load_files(
            image_dir, expand_dims
        )
        self.file_path = image_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            assert os.path.isdir(mask_dir)
            self.masks, _ = self._load_files(mask_dir, expand_dims)
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
        image_paths = phase_config["image_dir"]
        mask_paths = phase_config["mask_dir"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                image_paths[0],
                mask_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

    @abstractmethod
    def _load_files(self, dir, expand_dims):
        pass



class Standard_TIF_Dataset(Abstract_TIF_Dataset):
    """Dataset for tif files arranged in a file structure
    of multiple single image tifs located in a single 
    file with image and mask files located in differnt folders.
    e.g DSB2018, S_BIAD895, HeLaNuc

    Args:
        Abstract_TIF_Dataset (_type_): _description_
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
        )

    def _load_files(self, dir, expand_dims):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
            path = os.path.join(dir, file)
            if file.endswith(".tif"):
                img = np.asarray(imageio.imread(path))
            # check if file ends in ['.h5', '.hdf5']
            elif file.endswith(('.h5', '.hdf5')):
                with h5py.File(path, 'r') as f:
                    img = f['data'][:]
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths


class Hoechst_Dataset(Abstract_TIF_Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            pahse=phase,
            transformer_config=transformer_config,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
        )

    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
            path = os.path.join(dir, file)
            if file.endswith((".tif", ".png")):
                img = np.asarray(imageio.imread(path))
            elif file.endswith(('.h5', '.hdf5')):
                with h5py.File(path, 'r') as f:
                    img = f['data'][:]
            if img.ndim == 3:
                img = transforms.RgbToLabel()(img)
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths
    


class Tif_txt_Dataset(Abstract_TIF_Dataset):
    """Dataset for tif files arranged in a file structure
    of multiple single image tifs located in a single 
    file with image and mask files located in differnt folders.
    With a txt file containing the filenames of images split
    into train, val and test sets.
    e.g BBBC039, S_BIAD634 

    Args:
        Abstract_TIF_Dataset (_type_): _description_
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        filenames_path,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            filenames_path=filenames_path,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
        )
        

    def _load_files(self, dir, expand_dims):
        files_data = []
        paths = []
        for file in self.file_names:
            path = glob.glob(dir + f"/*{file}*")[0]
            if path.endswith((".tif", ".png")):
                img = np.asarray(imageio.imread(path))
            elif path.endswith(('.h5', '.hdf5')):
                with h5py.File(path, 'r') as f:
                    img = f['data'][:]
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