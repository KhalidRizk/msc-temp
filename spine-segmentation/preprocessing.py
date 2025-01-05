import os
import re
from pathlib import Path
from typing import Union

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from data_transforms.classes import *
from dataloader import VerSe, h5VerSe
from tqdm import tqdm
from utils.constants import COMPRESSION
from scipy import ndimage


def transform_to_str(transform):
    transform_str = transform.__class__.__name__ + ":\n"
    for key, value in transform.__dict__.items():
        transform_str += f"  {key}: {value}\n"
    return transform_str


class ImagePreprocessor:
    def preprocess(self, 
                   img: Union[np.ndarray, torch.Tensor, str, Path], 
                   msk: Union[np.ndarray, torch.Tensor, str, Path], 
                   transform: Union[tio.Compose, tio.Transform]) -> TioSubjectNoWarnings:

        if isinstance(img, (str, Path)):
            img_tio = tio.ScalarImage(img)
            msk_tio = tio.LabelMap(msk)
        else: 
            img_tio = tio.ScalarImage(tensor=img)
            msk_tio = tio.LabelMap(tensor=msk)
            
        subject_dict = {
            tio.IMAGE: img_tio,
            tio.LABEL: msk_tio,
        }

        subject = TioSubjectNoWarnings(subject_dict)
        subject = transform(subject)
        
        del img_tio
        del msk_tio
        
        return subject


class Preprocessing:
    def __init__(self, dataset_paths: str, output_name: str) -> None:
        output_base = 'datasets'

        output_name_full = output_name
        if '.h5' not in output_name:
            output_name_full = output_name + '.h5'

        self.output_dataset = Path(os.path.join(output_base, output_name, output_name_full))
        self.image_output = os.path.join(output_base, output_name)

        os.makedirs(self.image_output, exist_ok=True)

        self.dataset = dataset_paths
        self.img_preprocessor = ImagePreprocessor()

    def _get_split_name(self, path):
        split = (
            'training' if 'training' in str(path)
            else 'test' if 'test' in str(path)
            else 'validation'
        )
        return split

    def _create_split_group(self, split):
        with h5py.File(self.output_dataset, 'a') as f:
            if split in f:
                print(f"Failed to create group '{split}': group already exists.")
            else:
                f.create_group(split)

    def _sample_exists(self, split, sample):
        with h5py.File(self.output_dataset, 'a') as f:
            return sample in f[split]

    def _is_data_empty(self, split, sample):
        with h5py.File(self.output_dataset, 'a') as f:
            return (tio.IMAGE not in f[split][sample] or
                    tio.LABEL not in f[split][sample])

    def _delete_sample(self, split, sample):
        with h5py.File(self.output_dataset, 'a') as f:
            del f[split][sample]

    def _get_data_dtype(self, data):
        data_type_str = str(data.dtype)

        if 'float' in data_type_str:
            return np.float16
        elif 'int' in data_type_str:
            return np.int8
        else:
            return None
        
    def _process_sample(self, sample, img_data, split, transform, compression):
        img, msk, _ = img_data

        sample_exists = self._sample_exists(split, sample)

        if sample_exists:
            if self._is_data_empty(split, sample):
                print(f"Deleting empty sample '{sample}' from '{split}' split.")
                self._delete_sample(split, sample)
            else:
                print(f"Sample '{sample}' already exists in '{split}' split and is not empty. Skipping.")
                return

        subject = self.img_preprocessor.preprocess(img, msk, transform)

        image_data = subject[tio.IMAGE][tio.DATA].squeeze()
        mask_data = subject[tio.LABEL][tio.DATA].squeeze()

        with h5py.File(self.output_dataset, 'a') as f:
            split_group = f.require_group(split)
            sample_group = split_group.require_group(sample)
            
            img_dtype = self._get_data_dtype(image_data)
            msk_dtype = self._get_data_dtype(mask_data)
            sample_group.create_dataset(tio.IMAGE, data=image_data, compression=compression, dtype=img_dtype)
            sample_group.create_dataset(tio.LABEL, data=mask_data, compression=compression, dtype=msk_dtype)

        img_dir = os.path.join(self.image_output, split)
        os.makedirs(img_dir, exist_ok=True)
        subject.plot(show=False, output_path=os.path.join(img_dir, sample))
        plt.close()

        del subject

        del image_data
        del mask_data

    def preprocess_dataset(self, compression, transform) -> None:

        with open(os.path.join(os.path.dirname(self.output_dataset), 'transforms.txt'), "w") as file:
            for t in transform.transforms:
                file.write(transform_to_str(t) + "\n")

        for dataset_split in self.dataset:
            split = self._get_split_name(dataset_split[tio.IMAGE][0])
            print(f"Processing {split} ...")
            self._create_split_group(split)

            for img_data in (pbar := tqdm(list(zip(*dataset_split.values())))):
                filename = os.path.basename(img_data[0])
                matches = re.findall(r'verse\d+', filename)
                sample = '_'.join(matches)
                # if sample == 'verse006':
                #     continue
                pbar.set_description(f"Processing {sample}")
                self._process_sample(sample, img_data, split, transform, compression)



def dataset_preprocessing(
    out_name: str, 
    transform: tio.Transform, 
    dataset_dir: str = 'datasets', 
    edition: int = 19, 
    splits=('training', 'validation', 'test'),
    compression=COMPRESSION):
    
    datasets = {
        split: VerSe(root=dataset_dir, split=split, edition=edition, download=True)._load_paths() 
        for split in splits
    }

    preprocess = Preprocessing((datasets['training'], datasets['test'], datasets['validation']), output_name=out_name)
    preprocess.preprocess_dataset(compression=compression, transform=transform)


def morphological_cleanup(x):
    from scipy import ndimage
    x_np = x.numpy()
    x_np = ndimage.binary_closing(x_np)
    x_np = ndimage.binary_opening(x_np)
    return torch.from_numpy(x_np)

# def post_process_prediction(pred):
#     import torch
#     import numpy as np
#     from scipy import ndimage
#     from scipy.spatial import ConvexHull
#     from skimage.measure import regionprops, label

#     pred_np = pred.cpu().numpy()
#     pred_hull = np.zeros_like(pred_np)

#     for batch_idx in range(pred_np.shape[0]):  # Iterate over batch
#         for channel_idx in range(pred_np.shape[1]):  # Iterate over channels
#             spatial_data = pred_np[batch_idx, channel_idx]  # Extract spatial dimensions
#             labeled, num = ndimage.label(spatial_data > 0.5)
#             if num > 0:
#                 sizes = ndimage.sum(spatial_data > 0.5, labeled, range(1, num + 1))
#                 mask = sizes < (0.05 * sizes.max())
#                 remove_pixel = mask[labeled - 1]
#                 spatial_data[remove_pixel] = 0

#             spatial_data = ndimage.binary_fill_holes(spatial_data > 0.5)
#             labeled_hull, num_hull = label(spatial_data, return_num=True)

#             for region in regionprops(labeled_hull):
#                 coords = region.coords  # Extract coordinates for the region
#                 if len(coords) >= 3:
#                     hull = ConvexHull(coords)
#                     for simplex in hull.simplices:
#                         indices = coords[simplex].T
#                         spatial_data[tuple(indices)] = 1

#             pred_hull[batch_idx, channel_idx] = spatial_data

#     return torch.from_numpy(pred_hull).to(pred.device)

def post_process_prediction(pred):
    import torch
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    from skimage.measure import regionprops, label

    pred_np = pred.cpu().numpy()
    pred_hull = np.zeros_like(pred_np)

    for batch_idx in range(pred_np.shape[0]):  # Iterate over batch
        for channel_idx in range(pred_np.shape[1]):  # Iterate over channels
            spatial_data = pred_np[batch_idx, channel_idx]  # Extract spatial dimensions
            labeled, num = ndimage.label(spatial_data > 0.5)
            if num > 0:
                sizes = ndimage.sum(spatial_data > 0.5, labeled, range(1, num + 1))
                mask = sizes < (0.05 * sizes.max())
                remove_pixel = mask[labeled - 1]
                spatial_data[remove_pixel] = 0

            spatial_data = ndimage.binary_fill_holes(spatial_data > 0.5)
            labeled_hull, num_hull = label(spatial_data, return_num=True)

            for region in regionprops(labeled_hull):
                coords = region.coords  # Extract coordinates for the region
                if coords.shape[0] >= 3:  # At least 3 points required for ConvexHull
                    coords = coords[:, -3:]  # Use only spatial dimensions
                    if np.linalg.matrix_rank(coords) >= 2:  # Ensure points are at least 2D
                        try:
                            hull = ConvexHull(coords)
                            for simplex in hull.simplices:
                                indices = coords[simplex].T
                                spatial_data[tuple(indices)] = 1
                        except Exception as e:
                            # Skip region if ConvexHull fails
                            pass

            pred_hull[batch_idx, channel_idx] = spatial_data

    return torch.from_numpy(pred_hull).to(pred.device)


if __name__ == "__main__":
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.Clamp(-1024, 1024),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        MaskCutout(threshold=0.5),
        tio.Lambda(lambda x: morphological_cleanup(x))
    ])
    dataset_preprocessing(out_name='cutout_dataset', transform=transform)