import argparse
import logging
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.io import load_dataset
from src.models.io import (
    create_model,
    get_model_filename,
    load_model,
    read_model_config,
    save_model,
)

logger = logging.getLogger(__name__)


def main(
    model_name: str,
    dataset_name: str,
    epochs=10,
    batch_size=256,
    seed: int = 42,
    force_retrain: bool = False,
    continue_train: bool = False,
) -> None:
    logger.info(
        f"Running the training script for model '{model_name}' on dataset '{dataset_name}'."
    )

    # decide whether to modify the existing model or create a new one
    if path.exists(get_model_filename(model_name)):
        logger.info(f"The model '{model_name}' already exists.")
        if force_retrain:
            logger.warning(
                f"Training is running with 'force_retrain'. The existing file '{get_model_filename(model_name)}' will be overwritten during training."
            )
            model = create_model(model_name)
        elif continue_train:
            logger.warning(
                f"Training is running with 'continue_train'. The existing model '{get_model_filename(model_name)}' will be modified during training."
            )
            model = load_model(model_name)
        else:
            logger.info(
                "Stopping training. Consider setting 'force_retrain' or 'continue_train' to modify the existing model."
            )
            return
    else:
        model = create_model(model_name)

    # load dataset
    X, Y, _ = load_dataset(dataset_name)
    if not X.ndim == 3:
        raise ValueError(
            f"X must be of dimension (n_samples, n_timepoints, 4). Received {X.shape}."
        )
    if not Y.ndim == 2:
        raise ValueError(
            f"Y must be of dimension (n_samples, n_launch_params). Received {Y.shape}."
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # load model config
    model_config = read_model_config(model_name)
    model_type = model_config["model_type"]

    match model_type:
        case "MLP":
            X_train = torch.flatten(X_train, start_dim=1)
            X_test = torch.flatten(X_test, start_dim=1)
        case _:
            raise ValueError(f"The model type '{model_type}' is not supported!")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # create batches
    training_data = TensorDataset(X_train, y_train)
    loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        logger.info(f"Start epoch {epoch + 1}/{epochs}...")

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

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Evaluation
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)

    logger.info(f"Validation loss: {test_loss.item():.6f}")

    save_model(model, name=model_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to be trained.")
    parser.add_argument("dataset_name", help="Name of the training dataset.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default=10).",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default=256).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for the training (default=42).",
    )
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed

    main(model_name, dataset_name, epochs=epochs, batch_size=batch_size, seed=seed)
