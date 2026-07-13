import logging
import os

import torch
import torch.nn as nn
import yaml

from src.config import MODEL_CONFIG_DIR, MODEL_DIR
from src.models.MLP import MLP

logger = logging.getLogger(__name__)


def get_model_filename(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.pt")


def get_config_filename(name: str) -> str:
    return os.path.join(MODEL_CONFIG_DIR, f"{name}.yaml")


def save_model(model: nn.Module, name: str) -> None:
    outfile = get_model_filename(name)
    logger.info(f"Saving model to '{outfile}'...")
    torch.save(model.state_dict(), outfile)


def load_model(name: str) -> nn.Module:
    model_file = get_model_filename(name)
    logger.info(f"Loading model state_dict from '{model_file}'.")
    state_dict = torch.load(model_file)

    model = create_model(name)

    model.load_state_dict(state_dict)

    return model


def create_model(name: str) -> nn.Module:
    config_file = get_config_filename(name)
    config = read_model_config(name)
    if config == {}:
        raise ValueError(f"Could not create model '{name}'. No config file found.")

    model_type = config.get("model_type", "")
    if model_type == "":
        raise ValueError(
            f"Could not create model. No model_type specified in '{config_file}'."
        )

    match model_type:
        case "MLP":
            # load hyperparameters
            input_dim: int | None = config.get("input_dim")
            output_dim: int | None = config.get("output_dim")
            hidden_dim: int | list[int] | None = config.get("hidden_dim")
            num_hidden_layers: int | None = config.get("num_hidden_layers")

            if any(
                x is None
                for x in (input_dim, output_dim, hidden_dim, num_hidden_layers)
            ):
                raise ValueError(f"Missing model hyperparameters in '{config_file}'.")

            # create model
            model = MLP(
                input_dim=input_dim,  # pyright: ignore[reportArgumentType]
                hidden_dim=hidden_dim,  # pyright: ignore[reportArgumentType]
                output_dim=output_dim,  # pyright: ignore[reportArgumentType]
                num_hidden_layers=num_hidden_layers,  # pyright: ignore[reportArgumentType]
            )
        case _:
            raise ValueError(f"The model type '{model_type}' is not supported!")

    return model


def read_model_config(name: str) -> dict:
    config_file = get_config_filename(name)
    try:
        logger.info(f"Reading model config from '{config_file}'.")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.warning(
            f"Reading the specified config failed with error: {e}. "
            "Returning empty config."
        )
        return {}
