import os
import time
import json
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm

# Type aliases for clarity
TensorOrTensors = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
Logs = Dict[str, float]


def get_auto_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_sequence(x: Any) -> bool:
    """Returns True if x is a list or tuple."""
    return isinstance(x, (list, tuple))

def maybe_unpack(x: torch.Tensor, as_numpy: bool) -> Any:
    """Converts tensor to numpy if requested."""
    return x.cpu().numpy() if as_numpy else x

def validate_batch_input(batch: Any, with_labels: bool = False) -> Any:
    """If 'with_labels' is True, assume batch is a tuple (inputs, labels)."""
    return batch[0] if with_labels else batch

def validate_batch_input_output(batch: Any) -> Tuple[Any, Any]:
    """
    Expects batch to be a tuple (inputs, outputs).
    """
    return batch  # returns (xs, y)

class LossTracker:
    """A minimal loss tracker that averages batch losses over an epoch."""
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def __call__(self, loss: torch.Tensor) -> float:
        self.total += loss.item()
        self.count += 1
        return self.total / self.count

# -----------------------------
# Revised TrainableModule (fixed)
# -----------------------------
class TrainableModule(torch.nn.Module):
    """
    A simplified, Keras-like module for training with raw PyTorch.
    This version includes configuration, training, evaluation, prediction, 
    and weight saving/loading functionality.
    
    When an external module is provided via the `module` argument, it is wrapped.
    Otherwise, you can subclass TrainableModule and override the forward method.
    """
    def __init__(self, module: Optional[torch.nn.Module] = None):
        super().__init__()
        # Only wrap an external module if provided.
        if module is not None:
            self.module = module
            self.forward = self._call_wrapped_module
        # Note: Do NOT set self.module = self when module is None.
        self.device = get_auto_device()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.loss: Optional[Callable] = None
        self.metrics: Dict[str, Callable] = {}
        self.epoch: int = 0
        self.loss_tracker = LossTracker()

    def _call_wrapped_module(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward call for a wrapped module. Accepts variable arguments to support 
        multiple tensor inputs.
        """
        if len(inputs) == 1:
            return self.module(inputs[0])
        else:
            return self.module(*inputs)

    def configure(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        initial_epoch: int = 0,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "TrainableModule":
        """
        Set up training by specifying the optimizer, loss function, any metrics, and the device.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.metrics = metrics if metrics is not None else {}
        self.epoch = initial_epoch
        self.device = torch.device(device) if device is not None else get_auto_device()
        self.to(self.device)
        return self

    def unconfigure(self) -> "TrainableModule":
        """
        Clear training configuration (optimizer, scheduler, loss, metrics).
        Device remains unchanged.
        """
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.metrics = {}
        self.epoch = 0
        self.loss_tracker = LossTracker()
        return self

    def train_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """
        Perform one training step over the batch.
        """
        self.optimizer.zero_grad()
        # Forward pass – if xs is a sequence, pass as multiple arguments.
        y_pred = self(*xs) if is_sequence(xs) else self(xs)
        loss_value = self.loss(y_pred, y)
        loss_value.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        loss_log = self.loss_tracker(loss_value)
        metric_logs = {name: metric(y_pred, y) for name, metric in self.metrics.items()}
        return {"loss": loss_log, **metric_logs}

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        dev: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Logs:
        """
        Train the model on the provided dataloader.
        Optionally evaluate on a development set and invoke callbacks after each epoch.
        """
        callbacks = callbacks or []
        final_logs: Logs = {}
        total_epochs = self.epoch + epochs

        while self.epoch < total_epochs:
            self.train()
            self.loss_tracker.reset()
            for metric in self.metrics.values():
                if hasattr(metric, "reset"):
                    metric.reset()

            epoch_logs: Logs = {}
            pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
            for batch in pbar:
                xs, y = validate_batch_input_output(batch)
                if is_sequence(xs):
                    xs = tuple(x.to(self.device) for x in xs)
                else:
                    xs = xs.to(self.device)
                if is_sequence(y):
                    y = tuple(yi.to(self.device) for yi in y)
                else:
                    y = y.to(self.device)
                step_logs = self.train_step(xs, y)
                pbar.set_postfix(step_logs)
                epoch_logs = step_logs
            self.epoch += 1
            if dev is not None:
                eval_logs = self.evaluate(dev)
                epoch_logs.update(eval_logs)
            for callback in callbacks:
                callback(self, self.epoch, epoch_logs)
            final_logs = epoch_logs
        self.eval()
        return final_logs

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Logs:
        """
        Evaluate the model on the provided dataloader (no gradient updates).
        """
        self.eval()
        self.loss_tracker.reset()
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()
        logs: Logs = {}
        with torch.no_grad():
            for batch in dataloader:
                xs, y = validate_batch_input_output(batch)
                if is_sequence(xs):
                    xs = tuple(x.to(self.device) for x in xs)
                else:
                    xs = xs.to(self.device)
                if is_sequence(y):
                    y = tuple(yi.to(self.device) for yi in y)
                else:
                    y = y.to(self.device)
                y_pred = self(*xs) if is_sequence(xs) else self(xs)
                loss_value = self.loss(y_pred, y)
                loss_log = self.loss_tracker(loss_value)
                metric_logs = {name: metric(y_pred, y) for name, metric in self.metrics.items()}
                logs = {"loss": loss_log, **metric_logs}
        return logs

    def predict(self, dataloader: torch.utils.data.DataLoader, as_numpy: bool = True) -> List[Any]:
        """
        Compute predictions on the given dataloader.
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                xs = validate_batch_input(batch)
                if is_sequence(xs):
                    xs = tuple(x.to(self.device) for x in xs)
                else:
                    xs = xs.to(self.device)
                y = self.predict_step(xs, as_numpy=as_numpy)
                predictions.append(y)
        return predictions

    def predict_step(self, xs: TensorOrTensors, as_numpy: bool = True) -> Any:
        """
        Compute predictions for one batch. If xs is a tuple of (inputs, labels),
        only the inputs are used.
        """
        with torch.no_grad():
            if is_sequence(xs) and len(xs) == 2:
                x_input = xs[0]
            else:
                x_input = xs
            y = self(x_input)
            if is_sequence(y):
                return tuple(maybe_unpack(yi, as_numpy) for yi in y)
            else:
                return maybe_unpack(y, as_numpy)

    def save_weights(self, path: str) -> None:
        """
        Save the model's weights (its state_dict) to the given path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: Optional[Union[torch.device, str]] = None) -> None:
        """
        Load model weights from the given file.
        """
        if device is not None:
            self.device = torch.device(device)
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
