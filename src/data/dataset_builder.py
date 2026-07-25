import logging
from os import path

import torch
from tqdm import tqdm

import src.data.transforms as tf
from src.config import SIM_REPO_DIR
from src.data.repository import DataRepository
from src.data.sample import ProcessedSample, RawSample

logger = logging.getLogger(__name__)

TRANSFORMS = {
    # initial slicing
    "Resample": tf.Resample,
    "IndexSlice": tf.IndexSlice,
    "TimeSlice": tf.TimeSlice,
    # time series noise
    "GaussianNoise": tf.GaussianNoise,
    "RandomWalkNoise": tf.RandomWalkNoise,
}


class Pipeline:
    def __init__(self, transform_config: dict) -> None:
        # read transformations from config and initialize transformation classes
        self.transforms = []
        for transf in transform_config:
            name = list(transf.keys())[0]
            kwargs = transf[name]
            cls = TRANSFORMS[name]
            transform = cls(**kwargs)
            self.transforms.append(transform)

    def __call__(
        self, sample: RawSample, include_metadata: bool = False
    ) -> ProcessedSample:
        # create new processed sample from raw sample
        x = ProcessedSample.from_raw(sample, include_metadata=include_metadata)

        # apply transformations
        for transform in self.transforms:
            x = transform(x)

        return x


class DatasetBuilder:
    def __init__(self, repo_name: str, data_config: dict) -> None:
        # setup repo
        self.repo: str = path.join(SIM_REPO_DIR, f"{repo_name}.h5")

        # read config
        self.pipeline = Pipeline(data_config["transforms"])
        self.ts_features = data_config["ts_features"]
        self.targets = data_config["targets"]
        self.n_timepoints = data_config["n_timepoints"]

    def build_dataset(
        self, n_samples: int, verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Building dataset of {n_samples} samples...")
        n_ts_features = len(self.ts_features)
        n_targets = len(self.targets)

        # Initialize the feature and target tensors
        ts_features = torch.empty(
            n_samples, n_ts_features, self.n_timepoints, dtype=torch.float32
        )
        logger.info(f"Initialized feature tensor of shape {ts_features.shape}.")
        targets = torch.empty(n_samples, n_targets, dtype=torch.float32)
        logger.info(f"Initialized target tensor of shape {targets.shape}.")

        # fetch raw samples
        logger.info(f"Loading raw samples from '{self.repo}'.")
        with DataRepository(self.repo, "r") as repo:
            raw_samples = repo[:n_samples]

        # process raw samples
        logger.info("Processing raw samples...")
        iterator = enumerate(raw_samples)  # pyright: ignore[reportArgumentType]
        if verbose:
            iterator = tqdm(iterator)
        for i, sample in iterator:
            processed_sample = self.pipeline(sample)
            for j, feature in enumerate(self.ts_features):
                ts_features[i, j, :] = torch.from_numpy(
                    processed_sample.trajectory[feature]
                )
            for j, target in enumerate(self.targets):
                targets[i, j] = torch.from_numpy(processed_sample.targets[target])
        logger.info("Done!")

        return ts_features, targets
