import torch
import torch.nn as nn

import inspect
def debug_tensor_shape(tensor: torch.Tensor, tensor_name: str = "tensor"):
    """
    Debugs and prints the shape of a tensor along with the function and file name.
    
    Args:
        tensor (torch.Tensor): The tensor to debug.
        tensor_name (str): Name of the tensor for identification.
    """
    current_function = inspect.currentframe().f_back.f_code.co_name
    file_name = inspect.getfile(inspect.currentframe())
    print(f"[DEBUG] {tensor_name}.shape: {tensor.shape} | Function: {current_function} | File: {file_name}")

class MetricHandler:
    def __init__(self):
        self.metrics = {}

    def reset(self):
        for name in self.metrics:
            self.metrics[name]['total'] = 0.0
            try:
                self.metrics[name]['metric_fn']._loss.reset()
            except Exception as e:
                # print(e)
                pass

    def add_metric(self, name: str, metric_fn: nn.Module, is_accuracy: bool = False, weight: float = 1.0) -> None:
        self.metrics[name] = {
            'metric_fn': metric_fn,
            'is_accuracy': is_accuracy,
            'weight': weight,
            'total': 0.0
        }

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, accumulate_loss: bool = False) -> float:
        total_loss = 0.0 if accumulate_loss else None
        #debug_tensor_shape(outputs, "outputs")
        #debug_tensor_shape(targets, "targets")
        
        for name, metric in self.metrics.items():
            is_accuracy = metric['is_accuracy']
            metric_fn = metric['metric_fn']
            weight = metric['weight']

            if is_accuracy:
                with torch.no_grad():
                    value = metric_fn(outputs, targets)
            else:
                value = metric_fn(outputs, targets) * weight
                if accumulate_loss:
                    total_loss += value

            metric['total'] += value.item()
        return total_loss

    def compute(self, name: str, dataset_size: int) -> None:
        if name in self.metrics:
            return self.metrics[name]['total'] / dataset_size
        else:
            raise KeyError(f"Metric {name} not found.")

    def compute_metrics(self, dataset_size: int) -> dict:
        return {name: self.compute(name, dataset_size) for name in self.metrics}