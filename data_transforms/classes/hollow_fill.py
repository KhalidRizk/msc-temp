# data_transforms/classes/vertebrae_fill.py

import os
import h5py
import torch
import torchio as tio
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image
from utils.constants import COMPRESSION


class VertebraeFill(tio.SpatialTransform):
    def __init__(self, fill_holes=True, use_hull=True, **kwargs):
        super().__init__(**kwargs)
        self.fill_holes = fill_holes
        self.use_hull = use_hull
        self.temp_h5_path = 'temp_vertebrae_fill.h5'

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            if tio.LABEL in image_name:
                tensor = image[tio.DATA]
                processed = self._process_vertebrae(tensor)
                image.set_data(processed)
        return subject

    def _process_vertebrae(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
            
        unique_classes = torch.unique(tensor)
        unique_classes = unique_classes[unique_classes != 0]
        output_tensor = torch.zeros_like(tensor, dtype=torch.uint8)
        
        with h5py.File(self.temp_h5_path, 'w') as h5f:
            for class_value in unique_classes:
                class_mask = (tensor == class_value).float()
                h5f.create_dataset(
                    f'class_{int(class_value.item())}',
                    data=class_mask.squeeze().numpy(),
                    compression=COMPRESSION
                )
        
        with h5py.File(self.temp_h5_path, 'r') as h5f:
            for class_value in unique_classes:
                class_mask = np.array(h5f[f'class_{int(class_value.item())}'][:])
                
                if self.fill_holes:
                    class_mask = binary_fill_holes(class_mask)
                if self.use_hull:
                    class_mask = convex_hull_image(class_mask)
                
                class_tensor = torch.from_numpy(class_mask).to(tensor.device)
                output_tensor[class_tensor] = class_value
        
        os.remove(self.temp_h5_path)
        return output_tensor.squeeze(0)