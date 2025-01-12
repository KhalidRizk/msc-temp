import torch
import torchio as tio
from monai.transforms import FillHoles as MonaiFillHoles


class FillHoles(tio.LabelTransform):
    """TorchIO transform wrapper for MONAI's FillHoles transform
    
    This transform fills holes in binary masks. A hole is a background region 
    that is completely enclosed by foreground pixels.
    
    Args:
        **kwargs: Additional arguments to be passed to the TorchIO Transform base class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.monai_fill_holes = MonaiFillHoles()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Process only label images
        for image in self.get_images(subject):
            if not isinstance(image, tio.LabelMap):
                continue

            # Get image data
            label_data = image[tio.DATA]
            
            # Ensure binary
            if not torch.all(torch.logical_or(label_data == 0, label_data == 1)):
                label_data = (label_data > 0).float()
            
            # Apply MONAI's fill holes
            filled_data = self.monai_fill_holes(label_data)
            
            # Update the image data
            image.set_data(filled_data)

        return subject

    def is_invertible(self):
        return False