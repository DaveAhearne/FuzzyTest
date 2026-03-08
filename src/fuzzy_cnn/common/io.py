import torch
from pathlib import Path
from torch import nn
from torch.optim import Optimizer

# Utility functions for saving and loading model info from checkpoints, 
# we need the optimizer because some funcs like ADAM track gradients, so if we didn't
# have these saved then we'd be starting from zero every time

# Same with the epoch, if we wanna restart training it's good to have so we don't start 
# from the beginning

def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )

def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None, device: torch.device | str = "cpu") -> int:
    f = torch.load(path, map_location=device)

    model.load_state_dict(f["model"])

    if optimizer is not None:
        optimizer.load_state_dict(f["optimizer"])

    return f["epoch"]