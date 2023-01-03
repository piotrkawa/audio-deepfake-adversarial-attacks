import torch


def to_minmax(batch_x):
    mn, _ = torch.min(batch_x, dim=1, keepdim=True)
    mx, _ = torch.max(batch_x, dim=1, keepdim=True)

    r = mx - mn
    return (batch_x - mn) / r, mn, mx


def revert_minmax(batch_x, mn, mx):
    r = mx - mn
    return (batch_x * r) + mn
