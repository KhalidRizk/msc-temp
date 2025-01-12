import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from networks import *
from training.scripts import *
from prodigyopt import Prodigy
from torch import optim
import argparse
from utils.constants import DEVICE, MODELS_PATH
# from utils.visualize import *


def visualize_spine_segmentation(subject: tio.Subject, output_path: str, model: nn.Module, show: bool = False, threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        inputs = subject[tio.IMAGE][tio.DATA]
        device = next(model.parameters()).device
        inputs = inputs.unsqueeze(0).to(device)
        outputs = model(inputs)
        
        inputs = inputs.squeeze(0).cpu()
        outputs = outputs.squeeze(0).cpu()
        outputs = (outputs > threshold).float()

        # Create visualization subject with proper structure
        viz_subject = tio.Subject(
            t1=tio.ScalarImage(tensor=inputs),  # Using t1 as in training
            label=tio.LabelMap(tensor=outputs)  # Using label as in training
        )
        viz_subject.plot(output_path=output_path, show=show)
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='Spine Segmentation Inference')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the input nifti file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model for loading')
    parser.add_argument('--model', type=str, default='UNet', help='Model architecture')
    parser.add_argument('--input_shape', type=int, nargs=3, default=(64, 64, 128), help='Input shape for the model')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--output_path', type=str, default='prediction.nii.gz', help='Path to save prediction')
    parser.add_argument('--plot_path', type=str, default='middle_slice_visualization.png', help='Path to save visualization')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--num_classes', type=int, default=2)
    return parser.parse_args()


class SpineSegmentationInference:
    def __init__(self, args, device=DEVICE):
        self.device = device
        self.input_shape = args.input_shape
        self.overlap = (self.input_shape[0]//2, self.input_shape[1]//2, self.input_shape[2]//2)
        self.model, self.optimizer, self.scheduler = self._initialize_model(args)
        
        best_model = next((f for f in os.listdir(os.path.join(MODELS_PATH, args.model_name)) if 'BEST' in f), None)
        if best_model is None:
            raise FileNotFoundError(f"No best model found in {os.path.join(MODELS_PATH, args.model_name)}")
            
        load_model(self.model, self.optimizer, self.scheduler, 
                  model_path=os.path.join(MODELS_PATH, args.model_name, best_model))

        self.model.to(self.device)
        self.model.eval()

    def _initialize_model(self, args):
        model = get_model(args)
        optimizer = Prodigy(params=model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        return model, optimizer, scheduler

    def preprocess_scan(self, nifti_img):
        reoriented = nib.as_closest_canonical(nifti_img)
        scan_data = reoriented.get_fdata()
        scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())
        scan_tensor = torch.from_numpy(scan_data).float()
        if scan_tensor.ndim == 3:
            scan_tensor = scan_tensor.unsqueeze(0)
        return tio.Subject({
            tio.IMAGE: tio.ScalarImage(tensor=scan_tensor, affine=reoriented.affine)
        })


    def predict(self, nifti_path, output_path=None, plot_path=None, threshold=0.5):
        nifti_img = nib.load(nifti_path)
        
        with torch.no_grad():
            subject = self.preprocess_scan(nifti_img)
            grid_sampler = tio.inference.GridSampler(subject, self.input_shape, self.overlap)
            patch_loader = DataLoader(grid_sampler, batch_size=1)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
            
            for patches_batch in patch_loader:
                input_tensor = patches_batch[tio.IMAGE][tio.DATA].to(self.device)
                locations = patches_batch[tio.LOCATION]
                outputs = self.model(input_tensor)
                aggregator.add_batch(outputs, locations)
            
            output_tensor = F.sigmoid(aggregator.get_output_tensor())
            prediction = (output_tensor > threshold).float()
            prediction_nifti = nib.Nifti1Image(prediction.cpu().numpy().squeeze(), nifti_img.affine)
            
            if output_path:
                nib.save(prediction_nifti, output_path)
            
            if plot_path:
                subject[tio.LABEL] = tio.LabelMap(tensor=prediction)
                visualize_spine_segmentation(subject, plot_path, self.model, show=False, threshold=threshold)
            
            return prediction_nifti

if __name__ == "__main__":
    args = get_args()
    model_path = "/teamspace/studios/this_studio/spine-segmentation-binary/trained_models/AttentionUNet3D/BEST_AttentionUNet3D_epochs_127.pth"
    inferencer = SpineSegmentationInference(args)
    
    print(f"Initializing inference with model: {args.model_name}")
    
    inferencer = SpineSegmentationInference(args)
    
    prediction = inferencer.predict(
        nifti_path="/teamspace/studios/this_studio/verse/verse19-processed/dataset-01training/rawdata/sub-verse004/sub-verse004_ct_processed.nii.gz",
        output_path="prediction.nii.gz",
        plot_path="sub-verse004.png"
    )
    print("Inference complete!")