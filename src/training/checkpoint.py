import logging
import re
from os import path
from pathlib import Path

import torch
import torch.optim as opt

import src.models as mods
from src.utils import get_nested

logger = logging.getLogger(__name__)

MODELS = {
    "MLP": mods.MLP,
}

OPTIMIZERS = {
    "Adam": opt.Adam,
    "AdamW": opt.AdamW,
    "SGD": opt.SGD,
    "RMSprop": opt.RMSprop,
}

SCHEDULERS = {
    "StepLR": opt.lr_scheduler.StepLR,
    "MultiStepLR": opt.lr_scheduler.MultiStepLR,
    "CosineAnnealingLR": opt.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": opt.lr_scheduler.ReduceLROnPlateau,
}


def create_checkpoint(model, optimizer, scheduler, loss_history, epoch, index) -> dict:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "loss_history": loss_history,
        "epoch": epoch,
        "index": index,
    }
    return checkpoint


def unpack_checkpoint(checkpoint: dict, config: dict):
    # load model
    model_name = get_nested(config, "model", "name")
    if model_name is None:
        raise ValueError("No model specified in the training configuration!")
    model_params = get_nested(config, "model", "params", default={})
    model_class = MODELS.get(model_name)  # pyright: ignore[reportArgumentType]
    model = model_class(**model_params)  # pyright: ignore[reportOptionalCall]
    model_state_dict = checkpoint.get("model_state")
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    # load optimizer
    optim_name = get_nested(config, "optimizer", "name")
    if optim_name is None:
        raise ValueError("No optimizer specified in the training configuration!")
    optim_params = get_nested(config, "optimizer", "params", default={})
    optim_class = OPTIMIZERS.get(optim_name)  # pyright: ignore[reportArgumentType]
    optimizer = optim_class(**optim_params)  # pyright: ignore[reportOptionalCall]
    optimizer_state_dict = checkpoint.get("optimizer_state")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # load scheduler
    scheduler_name = get_nested(config, "scheduler", "name")
    if scheduler_name is None:
        scheduler = None
    else:
        scheduler_params = get_nested(config, "scheduler", "params", default={})
        scheduler_class = SCHEDULERS.get(scheduler_name)  # pyright: ignore[reportArgumentType]
        scheduler = scheduler_class(**scheduler_params)  # pyright: ignore[reportOptionalCall]
        scheduler_state_dict = checkpoint.get("scheduler_state")
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

    # load rest
    loss_history = checkpoint.get("loss_history", {"train_loss": [], "test_loss": []})
    epoch = checkpoint.get("epoch", 0)
    index = checkpoint.get("index", 0)

    return model, optimizer, scheduler, loss_history, epoch, index


def save_checkpoint(checkpoint: dict, directory: str, name: str | None = None) -> None:
    if name is not None:
        filename = path.join(directory, f"{name}.pt")
    else:
        filename = path.join(directory, f"checkpoint_{checkpoint['index']:04}.pt")
    logger.info(f"Saving checkpoint '{filename}'...")
    torch.save(checkpoint, filename)


def load_checkpoint(
    directory: str, index: int | None = None, name: str | None = None
) -> dict:
    if name is None and index is None:
        raise ValueError("Either checkpoint name or index must be specified!")
    if name is not None:
        filename = path.join(directory, f"{name}.pt")
    else:
        filename = path.join(directory, f"checkpoint_{index:04}.pt")

    logger.info(f"Loading checkpoint '{filename}'...")
    checkpoint = torch.load(filename)

    return checkpoint


def get_latest_checkpoint_number(directory):
    pattern = re.compile(r"checkpoint_(\d{4})\.pt")
    checkpoints = []
    for file in Path(directory).iterdir():
        match = pattern.fullmatch(file.name)
        if match:
            checkpoints.append(int(match.group(1)))

    if checkpoints:
        return max(checkpoints)
    else:
        return -1


def delete_checkpoints_after(directory, max_checkpoint):
    logger.info(
        f"Deleting all checkpoints in directory '{directory}' with index greater than {max_checkpoint}."
    )
    pattern = re.compile(r"checkpoint_(\d{4})\.pt")
    for file in Path(directory).iterdir():
        match = pattern.fullmatch(file.name)
        if match:
            checkpoint_num = int(match.group(1))
            if checkpoint_num > max_checkpoint:
                file.unlink()
                logger.debug(f"Deleted '{file}'.")
    logger.info("Done!")
