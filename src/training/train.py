import argparse
import logging
from datetime import datetime
from os import path
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import LOGS_DIR
from src.data.io import load_dataset
from src.training.io import (
    create_checkpoint,
    delete_checkpoints_after,
    get_latest_checkpoint_number,
    load_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)
from src.utils import get_nested, read_config

logger = logging.getLogger(__name__)


LOSS_FUNCS = {
    "MSE": nn.MSELoss,
    "MAE": nn.L1Loss,
}


def main(
    config_file: str,
    dataset_name: str,
    epochs: int,
    check_freq: int,
    start_checkpoint: int | None = None,
) -> None:
    directory = path.dirname(config_file)
    config = read_config(config_file)

    # load checkpoint
    latest_checkpoint = get_latest_checkpoint_number(directory)
    if start_checkpoint is None:
        start_checkpoint = latest_checkpoint
    else:
        if start_checkpoint > latest_checkpoint:
            raise ValueError(
                "The 'start_checkpoint' is greater than the latest checkpoint "
                f"({start_checkpoint} vs {latest_checkpoint})."
            )
    delete_checkpoints_after(directory, start_checkpoint)

    # instantiate stuff
    if start_checkpoint >= 0:
        checkpoint = load_checkpoint(directory, index=start_checkpoint)
    else:
        checkpoint = {}
    model, optimizer, scheduler, loss_history, start_epoch, index = unpack_checkpoint(
        checkpoint, config
    )

    # load data
    X, Y, Z = load_dataset(dataset_name)
    if not X.ndim == 3:
        raise ValueError(
            f"X must be of dimension (n_samples, n_timepoints, 4). Received {X.shape}."
        )
    if not Y.ndim == 2:
        raise ValueError(
            f"Y must be of dimension (n_samples, n_launch_params). Received {Y.shape}."
        )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=get_nested(config, "training", "test_size", default=0.2),
        random_state=get_nested(config, "training", "seed", default=42),
    )

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # create batches
    training_data = TensorDataset(X_train, y_train)
    loader = DataLoader(
        training_data,
        batch_size=get_nested(config, "training", "batch_size", default=256),  # pyright: ignore[reportArgumentType]
        shuffle=True,
    )

    # set up loss function
    loss_name = get_nested(config, "loss", "name")
    if loss_name is None:
        raise ValueError("No loss function specified in the training configuration!")
    loss_params = get_nested(config, "loss", "params", default={})
    loss_class = LOSS_FUNCS.get(loss_name)  # pyright: ignore[reportArgumentType]
    criterion = loss_class(**loss_params)  # pyright: ignore[reportOptionalCall]

    # run training loop
    logger.info(
        f"Starting training with loss function '{loss_name}' and optimizer '{optimizer}'..."
    )
    end_epoch = start_epoch + epochs
    for epoch in range(start_epoch, end_epoch):
        logger.info(f"Start epoch {epoch + 1}/{end_epoch}...")

        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in tqdm(loader):
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        loss_history["train_loss"].append(avg_loss)

        logger.info(f"Epoch {epoch + 1}/{end_epoch}, Loss: {avg_loss:.6f}")

        if epoch % check_freq == 0:
            index += 1
            checkpoint = create_checkpoint(
                model, optimizer, scheduler, loss_history, epoch, index
            )
            save_checkpoint(checkpoint, directory)

    # Evaluation
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)

    logger.info(f"Validation loss: {test_loss.item():.6f}")


if __name__ == "__main__":
    # logger setup
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logfile = path.join(LOGS_DIR, f"{timestamp}_{script_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile),
        ],
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of the training configuration.")
    parser.add_argument("dataset_name", help="Name of the training dataset.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default=10).",
    )
    parser.add_argument(
        "-f",
        "--frequency-checkpoint",
        type=int,
        default=5,
        help="Number of epochs after which a checkpoint is saved (default=5).",
    )
    parser.add_argument(
        "-s",
        "--start-checkpoint",
        type=int,
        help="Checkpoint index from which to continue training (default=latest_checkpoint).",
    )
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    config_name = args.config_name
    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed

    main(
        args.config_file,
        args.dataset_name,
        epochs=args.epochs,
        check_freq=args.frequency_checkpoint,
        start_checkpoint=args.start_checkpoint,
    )
