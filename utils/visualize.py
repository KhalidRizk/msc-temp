import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from monai.networks.utils import predict_segmentation
from torch.utils.data import Dataset
from utils.constants import HEATMAP
from utils.misc import combine_bboxes_image, compute_bbox3d, create_bbox_image


def visualize_spine_localization_heatmap(subject: tio.Subject, output_path: str, model: nn.Module, show: bool = False):
    model.eval()
    with torch.no_grad():
        inputs, targets = subject[tio.IMAGE][tio.DATA], subject[tio.LABEL][tio.DATA]

        device = next(model.parameters()).device
        inputs = inputs.unsqueeze(0).to(device)

        outputs = model(inputs)

        inputs = inputs.squeeze(0).cpu()
        outputs = outputs.squeeze(0).cpu()
        targets = targets.cpu()

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=inputs),
            target=tio.LabelMap(tensor=targets),
            prediction=tio.LabelMap(tensor=outputs)
        )

        subject.plot(output_path=output_path, show=show)
        plt.close()


def visualize_spine_localization_heatmap_detailed(
    subject: tio.Subject,
    output_path: str,
    model: nn.Module = None,
    show: bool = False,
    act: nn.Module = F.softmax,
):
    if model:
        model.eval()
    with torch.no_grad():
        inputs, label = subject[tio.IMAGE][tio.DATA], subject[tio.LABEL][tio.DATA]
        device = next(model.parameters()).device if model else torch.device("cpu")

        # print(f"Labels Unique: {torch.unique(label)}")
        # print(f"Labels Shape: {label.shape}")

        outputs = model(inputs.unsqueeze(0).to(device)).squeeze(0).cpu() if model else torch.zeros_like(label)
        outputs = act(outputs, dim=1) 
        inputs = inputs.cpu()

        # print(f"Outputs Unique: {torch.unique(outputs)}")
        # print(f"Outputs Shape: {outputs.shape}")

        heatmap = subject.get(HEATMAP, {"data": torch.zeros_like(label)})[tio.DATA].squeeze(0).cpu()
        # print(f"Heatmap Shape: {heatmap.shape}")
        # print(f"Heatmap Unique: {torch.unique(heatmap)}")

        vertebrae_bboxes = []
        for channel_idx in range(outputs.shape[0]):
            vertebra_heatmap = outputs[channel_idx] 

            if torch.any(vertebra_heatmap):
                bbox = compute_bbox3d(vertebra_heatmap)
                vertebrae_bboxes.append(bbox)
            else:
                vertebrae_bboxes.append(None)

        visualization_image = torch.zeros(subject.spatial_shape, dtype=torch.uint8)

        for channel_idx, bbox in enumerate(vertebrae_bboxes):
            if bbox is not None:
                bbox_image = create_bbox_image(bbox, subject.spatial_shape, thickness=2)
                visualization_image = torch.where(bbox_image == 1, channel_idx + 1, visualization_image)

        combined_bboxes = combine_bboxes_image([visualization_image.unsqueeze(0)])
        # print(f"Combined BBoxes Shape: {combined_bboxes.shape}")

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=inputs),
            label=tio.LabelMap(tensor=label),
            bboxes=tio.LabelMap(tensor=combined_bboxes),
        )
        
        if torch.any(heatmap):
            subject.add_image(tio.LabelMap(tensor=heatmap), HEATMAP)

        if model:
            subject.add_image(tio.LabelMap(tensor=outputs), "Predicted heatmap")

        # subject.plot(output_path=output_path, show=show)
        subject.plot(output_path=output_path, show=False)

        fig = plt.gcf()
        axes = fig.get_axes()

        # Modify the titles
        for ax in axes:
            title = ax.get_title()
            if title:
                # Split the title into two lines or make it smaller
                ax.set_title(title, fontsize=8, wrap=True)  # Adjust fontsize and wrap as needed

        # Save or display the updated plot
        if output_path:
            plt.savefig(output_path)
        if show:
            plt.show()
        # plt.close()
        plt.close()


def visualize_spine_segmentation(subject: tio.Subject, output_path: str, model: nn.Module, show: bool = False, threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        inputs, targets = subject[tio.IMAGE][tio.DATA], subject[tio.LABEL][tio.DATA]

        device = next(model.parameters()).device
        inputs = inputs.unsqueeze(0).to(device)

        outputs = model(inputs)

        inputs = inputs.squeeze(0).cpu()
        outputs = outputs.squeeze(0).cpu()
        targets = targets.cpu()

        outputs = (outputs > threshold).float()

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=inputs),
            target=tio.LabelMap(tensor=targets),
            prediction=tio.LabelMap(tensor=outputs)
        )

        subject.plot(output_path=output_path, show=show)
        plt.close()


def visualize_vertebrae_segmentation(subject: tio.Subject, output_path: str, model: nn.Module, show: bool = False): # unused
    model.eval()
    with torch.no_grad():

        inputs, targets = subject[tio.IMAGE][tio.DATA], subject[tio.LABEL][tio.DATA]
        id = subject['subject_id']

        device = next(model.parameters()).device
        inputs = inputs.unsqueeze(0).to(device)
        outputs = model(inputs)

        outputs = predict_segmentation(logits=outputs, mutually_exclusive=True)

        inputs = inputs.squeeze(0).cpu()
        outputs = outputs.cpu()
        targets = targets.argmax(dim=0, keepdim=True).cpu()

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=inputs),
            target=tio.LabelMap(tensor=targets),
            prediction=tio.LabelMap(tensor=outputs.squeeze(0))
        )

        ins_classes = torch.unique(targets)[1:].tolist()
        preds_classes = torch.unique(outputs)[1:].tolist()

        ins_str = "_".join([str(int(cls)) for cls in ins_classes])
        preds_str = "_".join([str(cls) for cls in preds_classes])

        image_name = f"_{id}_input_class_{ins_str}_pred_classes_{preds_str}"

        output_parts = output_path.split('.')
        output_name = output_parts[0] + image_name + '.' + output_parts[1]

        subject.plot(output_path=output_name, show=show)
        plt.close()


def visualize_scalar(subject: tio.Subject, output_path: str, model: nn.Module, show: bool = False):
    model.eval()
    with torch.no_grad():
        inputs, targets = subject[tio.IMAGE][tio.DATA], subject[tio.LABEL][tio.DATA]

        device = next(model.parameters()).device
        inputs = inputs.unsqueeze(0).to(device)

        outputs = model(inputs)

        inputs = inputs.squeeze(0).cpu()
        outputs = outputs.squeeze(0).cpu()
        targets = targets.cpu()

        subject = tio.Subject(
            input=tio.ScalarImage(tensor=inputs),
            target=tio.ScalarImage(tensor=targets),
            prediction=tio.ScalarImage(tensor=outputs)
        )

        subject.plot(output_path=output_path, show=show)
        plt.close()


if __name__ == "__main__":
    ...