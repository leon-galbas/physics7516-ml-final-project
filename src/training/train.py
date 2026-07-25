import argparse
import logging
from datetime import datetime
from os import path
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import LOGS_DIR
from src.data.dataset_builder import DatasetBuilder
from src.simulation.generator import TrajectoryGenerator
from src.training.checkpoint import (
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
    epochs: int,
    check_freq: int,
    start_checkpoint: int | None = None,
) -> None:
    logger.info(f"Starting the experiment specified in '{config_file}'.")
    directory = path.dirname(config_file)
    config = read_config(config_file)

    # generate trajectories
    sim_config = config["simulation"]
    n_samples = sim_config["n_samples"]
    generator = TrajectoryGenerator(**sim_config)
    generator.generate_to_repo(sim_config["repo_name"], n_samples, verbose=True)

    # load data
    features_path = path.join(directory, "X.pt")
    targets_path = path.join(directory, "Y.pt")
    if path.exists(features_path) and path.exists(targets_path):
        logger.info("The processed dataset already exists. Loading data...")
        X = torch.load(features_path)
        logger.info(f"Loaded feature dataset of shape {X.shape}.")
        Y = torch.load(targets_path)
        logger.info(f"Loaded target dataset of shape {Y.shape}.")
    else:
        repo_name = get_nested(config, "simulation", "repo_name")
        builder = DatasetBuilder(repo_name, config["data"])  # pyright: ignore[reportArgumentType]
        X, Y = builder.build_dataset(n_samples, verbose=True)
        torch.save(X, features_path)
        torch.save(Y, targets_path)
    if X.size(0) != Y.size(0):
        raise ValueError(
            f"The number of samples in X and Y do not match ({X.size(0)} vs {Y.size(0)})."
        )

    # train/test split
    test_size = get_nested(config, "training", "test_size", default=0.2)
    seed = get_nested(config, "training", "seed", default=42)
    n = X.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))  # pyright: ignore[reportArgumentType]

    test_n = int(test_size * n)  # pyright: ignore[reportOperatorIssue]
    test_idx = perm[:test_n]
    train_idx = perm[test_n:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # create batches
    training_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(
        training_data,
        batch_size=get_nested(config, "training", "batch_size", default=256),  # pyright: ignore[reportArgumentType]
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=get_nested(config, "training", "batch_size", default=256),  # pyright: ignore[reportArgumentType]
        shuffle=False,
    )

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

        # Run training
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        loss_history["train_loss"].append(train_loss)

        # compute validation loss
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(test_loader):
                prediction = model(X_batch)
                loss = criterion(prediction, Y_batch)
                running_loss += loss.item()
        test_loss = running_loss / len(test_loader)
        loss_history["test_loss"].append(test_loss)

        logger.info(
            f"Epoch {epoch + 1}/{end_epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
        )

        if epoch % check_freq == 0:
            index += 1
            checkpoint = create_checkpoint(
                model, optimizer, scheduler, loss_history, epoch, index
            )
            save_checkpoint(checkpoint, directory)


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

    main(
        args.config_file,
        epochs=args.epochs,
        check_freq=args.frequency_checkpoint,
        start_checkpoint=args.start_checkpoint,
    )
