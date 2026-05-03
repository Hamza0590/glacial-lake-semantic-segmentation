import torch
import config


class IoUMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_bin = (preds >= config.THRESHOLD).float()
        targets_bin = targets.float()
        self.tp += (preds_bin * targets_bin).sum().item()
        self.fp += (preds_bin * (1 - targets_bin)).sum().item()
        self.fn += ((1 - preds_bin) * targets_bin).sum().item()

    def compute(self) -> float:
        return self.tp / (self.tp + self.fp + self.fn + 1e-8)


class F1Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_bin = (preds >= config.THRESHOLD).float()
        targets_bin = targets.float()
        self.tp += (preds_bin * targets_bin).sum().item()
        self.fp += (preds_bin * (1 - targets_bin)).sum().item()
        self.fn += ((1 - preds_bin) * targets_bin).sum().item()

    def compute(self) -> float:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-8)


def compute_batch_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    preds_bin = (preds >= threshold).float()
    targets_bin = targets.float()
    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1 - targets_bin)).sum().item()
    fn = ((1 - preds_bin) * targets_bin).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return {"iou": iou, "f1": f1}
