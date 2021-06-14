import torch

def calc_acc(preds: torch.Tensor, labels: torch.Tensor):
    _, pred_max = torch.max(preds, 1)
    acc = torch.sum(pred_max == labels.data, dtype=torch.float64) / len(preds)
    return acc