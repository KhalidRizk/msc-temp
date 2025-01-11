import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from monai import losses
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric

''' for:

    - denoising:
            * losses: custom self constrained loss defined in train loop
            * eval: SSIM, PSNR, RMSE

    - heatmap:
            * losses: MSE / AdaptiveWing / Wing
            * eval: bbIoU, RMSE, R2, MAE

    - segmentation:
            * losses: BCE / Dice / DiceCE
            * eval: IoU (Jaccard), F1Score, Recall, Precision, Haussdorf
'''

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


###########################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss


###########################################################################
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, softmax=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class TFL2Loss(nn.Module):
    """
    Equivalent to the following TensorFlow loss function:
    def tensorflow_l2_loss(pred, target, batch_size):
        return tf.nn.l2_loss(pred - target) / batch_size
    Used in SCNet paper.
    """
    def __init__(self):
        super(TFL2Loss, self).__init__()

    def forward(self, pred, target):
        batch_size = pred.size(0)
        pred = F.sigmoid(pred)
        return F.mse_loss(pred, target, reduction='sum') / (2 * batch_size)


###########################################################################
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        return F.mse_loss(pred, target, reduction='sum')


###########################################################################
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = F.sigmoid(pred)
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


###########################################################################
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, reduction='mean'):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = F.sigmoid(pred)
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        total_loss = torch.cat([loss1, loss2])

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'none':
            return total_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


###########################################################################
class bbIoU(nn.Module):
    """ WARNING: code has been used only as evaluation metric for heatmap regression.
    Computes the Intersection over Union (IoU) for predicted and target bounding boxes.
    The bounding boxes are derived by thresholding the 3D input tensors.
    """
    def __init__(self, threshold=0.5):
        super(bbIoU, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        batch_size = pred.size(0)
        iou_sum = 0.0
        pred = F.softmax(pred, dim=1)
        for i in range(batch_size):
            pred_bbox = self.extract_bbox(pred[i] > self.threshold)
            target_bbox = self.extract_bbox(target[i] > self.threshold)

            iou_sum += self.calculate_iou(pred_bbox, target_bbox)

        return iou_sum / batch_size

    def extract_bbox(self, mask):
        # Assuming mask is a binary tensor [C, H, W, D]
        pos = torch.where(mask)
        if len(pos[0]) == 0:
            return torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32).to(mask.device)  # Return an empty box if no true values
        xmin, ymin, zmin = pos[1].min(), pos[2].min(), pos[3].min()
        xmax, ymax, zmax = pos[1].max(), pos[2].max(), pos[3].max()
        return torch.tensor([xmin, ymin, zmin, xmax, ymax, zmax], dtype=torch.float32).to(mask.device)

    def calculate_iou(self, bbox1, bbox2):
        x1 = torch.max(bbox1[0], bbox2[0]) # Calculate intersection coordinates
        y1 = torch.max(bbox1[1], bbox2[1])
        z1 = torch.max(bbox1[2], bbox2[2])
        x2 = torch.min(bbox1[3], bbox2[3])
        y2 = torch.min(bbox1[4], bbox2[4])
        z2 = torch.min(bbox1[5], bbox2[5])

        inter_vol = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0) * torch.clamp(z2 - z1, min=0) # Calculate intersection volume

        bbox1_vol = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2]) # Calculate volumes of the individual bounding boxes
        bbox2_vol = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])

        union_vol = bbox1_vol + bbox2_vol - inter_vol # Union volume

        return inter_vol / union_vol if union_vol > 0 else torch.tensor(0.0).to(bbox1.device) # Compute IoU


###########################################################################
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.SSIMLoss(spatial_dims=3, data_range=1, reduction='mean')

    def forward(self, predicted, target):
        predicted = F.sigmoid(predicted)
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torchmetrics.PeakSignalNoiseRatio(data_range=(0, 1), dim=1)

    def forward(self, predicted, target):
        predicted = F.sigmoid(predicted)
        loss = self._loss(predicted, target)
        if torch.isinf(loss):
            loss = torch.tensor(100.0) # capping for psnr
        return loss


###########################################################################
class RMSE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._loss = nn.MSELoss(reduction='mean')
        self.eps = eps

    def forward(self, predicted, target):
        predicted = F.softmax(predicted, dim=1)
        loss = torch.sqrt(self._loss(predicted, target) + self.eps)
        return loss


###########################################################################
class KLDiv(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, predicted, target):
        predicted = F.log_softmax(predicted, dim=1)
        target = F.softmax(target, dim=1)
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class F1Score(nn.Module):
    def __init__(self):
        super().__init__()
        self._f1 = torchmetrics.F1Score(task='multiclass', num_classes=26)

    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        target_labels = torch.argmax(target, dim=1)
        f1 = self._f1(predicted_probs, target_labels)
        return f1


###########################################################################
class Recall(nn.Module):
    def __init__(self):
        super().__init__()
        self._recall = torchmetrics.Recall(task='multiclass', num_classes=26)

    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        target_labels = torch.argmax(target, dim=1)
        recall = self._recall(predicted_probs, target_labels)
        return recall


###########################################################################
class Precision(nn.Module):
    def __init__(self):
        super().__init__()
        self._precision = torchmetrics.Precision(task='multiclass', num_classes=26)

    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        target_labels = torch.argmax(target, dim=1)
        precision = self._precision(predicted_probs, target_labels)
        return precision


###########################################################################
class Hausdorff(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.HausdorffDTLoss(sigmoid=True, reduction='mean')

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class Jaccard(nn.Module):
    def __init__(self):
        super().__init__()
        self._jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=26)

    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        target_labels = torch.argmax(target, dim=1)
        jaccard = self._jaccard(predicted_probs, target_labels)
        return jaccard


###########################################################################
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.L1Loss(reduction='mean')

    def forward(self, predicted, target):
        predicted = F.softmax(predicted, dim=1)
        loss = self._loss(predicted, target)
        return loss


###########################################################################
class R2(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torchmetrics.R2Score()

    def forward(self, predicted, target):
        predicted = F.softmax(predicted, dim=1)
        loss = self._loss(predicted.flatten(), target.flatten())
        return loss

###########################################################################
class IDRate(nn.Module):
    """
    ID Rate Loss specifically for vertebrae segmentation, where:
    - Not all vertebrae may be present in a single scan
    - Each vertebra should be identified correctly when present
    - Missing vertebrae should not affect the score
    
    The rate considers:
    1. Which vertebrae are actually present in the scan
    2. How accurately each present vertebra is identified
    3. Ignores vertebrae that are not in the current scan
    
    Returns a score between 0 and 1, where:
    - 1 means all present vertebrae were correctly identified
    - 0 means none of the present vertebrae were correctly identified
    """
    def __init__(self, num_classes=26):  # 25 vertebrae + background
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        predicted_labels = torch.argmax(predicted_probs, dim=1)
        target_labels = torch.argmax(target, dim=1)
        
        batch_size = predicted.size(0)
        batch_rates = []
        
        for b in range(batch_size):
            present_vertebrae = torch.unique(target_labels[b])
            present_vertebrae = present_vertebrae[present_vertebrae != 0]
            
            if len(present_vertebrae) == 0:
                continue
                
            vertebra_rates = []
            for vert_idx in present_vertebrae:
                target_mask = (target_labels[b] == vert_idx)
                if not target_mask.any():
                    continue
                    
                pred_mask = (predicted_labels[b] == vert_idx)
                correct_pixels = torch.logical_and(target_mask, pred_mask).sum()
                total_pixels = target_mask.sum()
                
                vert_rate = correct_pixels.float() / total_pixels.float()
                vertebra_rates.append(vert_rate)
            
            if vertebra_rates:
                scan_rate = torch.stack(vertebra_rates).mean()
                batch_rates.append(scan_rate)
        
        if not batch_rates:
            return torch.tensor(0.0, device=predicted.device)
            
        return torch.stack(batch_rates).mean()


###########################################################################
class MulticlassHausdorff(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.num_classes = num_classes
        self._metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95
        )

    def forward(self, predicted, target):
        predicted = F.softmax(predicted, dim=1)
        predicted_labels = torch.argmax(predicted, dim=1)
        target_labels = torch.argmax(target, dim=1)
        
        batch_size = predicted.size(0)
        hausdorff_distances = []
        
        for b in range(batch_size):
            class_distances = []
            present_classes = torch.unique(target_labels[b])
            
            for c in present_classes:
                pred_binary = (predicted_labels[b] == c).float()
                target_binary = (target_labels[b] == c).float()
                
                if pred_binary.ndim == 3:
                    pred_binary = pred_binary.unsqueeze(0).unsqueeze(1)
                    target_binary = target_binary.unsqueeze(0).unsqueeze(1)
                elif pred_binary.ndim == 4:
                    pred_binary = pred_binary.unsqueeze(1)
                    target_binary = target_binary.unsqueeze(1)
                
                distance = self._metric(pred_binary, target_binary)
                class_distances.append(distance)
            
            if class_distances:
                hausdorff_distances.append(torch.stack(class_distances).mean())
        
        if not hausdorff_distances:
            return torch.tensor(float('inf')).to(predicted.device)
            
        return torch.stack(hausdorff_distances).mean()