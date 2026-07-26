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
    # features
    "ComputeVelocities": tf.ComputeVelocities,
    "ComputeAccelerations": tf.ComputeAccelerations,
    "ComputeScalarFeatures": tf.ComputeScalarFeatures,
    # targets
    "ComputeHitpoint": tf.ComputeHitpoint,
    "ComputeEuclideanVelocity": tf.ComputeEuclideanVelocity,
}


class Pipeline:
    def __init__(self, transform_config: dict) -> None:
        # read transformations from config and initialize transformation classes
        self.transforms = []
        for transf in transform_config:
            if isinstance(transf, str):
                cls = TRANSFORMS[transf]
                transform = cls()
                self.transforms.append(transform)
            elif isinstance(transf, dict):
                name = list(transf.keys())[0]
                kwargs = transf[name]
                cls = TRANSFORMS[name]
                transform = cls(**kwargs)
                self.transforms.append(transform)
            else:
                raise TypeError(
                    "Transform specified in config must be of type 'str' or 'dict', "
                    f"got '{type(transf)}'."
                )

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
        self.n_timepoints = data_config.get("n_timepoints")
        self.feature_type = data_config.get("feature_type")
        if self.feature_type not in ["timeseries", "scalar"]:
            raise ValueError(
                "Feature type must be either 'timeseries' or 'scalar', "
                f"got '{self.feature_type}'."
            )
        if self.feature_type == "timeseries" and self.n_timepoints is None:
            raise ValueError(
                "If feature_type 'timeseries' is set, must provide 'n_timepoints' "
                "value in the data config."
            )
        self.pipeline = Pipeline(data_config["transforms"])
        self.features = data_config["features"]
        self.targets = data_config["targets"]

    def build_dataset(
        self, n_samples: int, verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Building dataset of {n_samples} samples...")
        n_features = len(self.features)
        n_targets = len(self.targets)

        # Initialize the feature and target tensors
        if self.feature_type == "timeseries":
            features = torch.empty(
                n_samples,
                n_features,
                self.n_timepoints,  # pyright: ignore[reportArgumentType]
                dtype=torch.float32,
            )
        else:
            features = torch.empty(n_samples, n_features, dtype=torch.float32)
        logger.info(f"Initialized feature tensor of shape {features.shape}.")
        targets = torch.empty(n_samples, n_targets, dtype=torch.float32)
        logger.info(f"Initialized target tensor of shape {targets.shape}.")

        # process raw samples
        logger.info(f"Processing raw samples from '{self.repo}'...")
        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator, total=n_samples, desc="Processing")  # pyright: ignore[reportArgumentType]
        for i in iterator:
            with DataRepository(self.repo, "r") as repo:
                raw_sample = repo[i]
            processed_sample = self.pipeline(raw_sample)  # pyright: ignore[reportArgumentType]
            for j, feature in enumerate(self.features):
                if self.feature_type == "timeseries":
                    features[i, j, :] = torch.from_numpy(
                        processed_sample.trajectory[feature]
                    )
                else:
                    features[i, j] = processed_sample.scalars[feature]
            for j, target in enumerate(self.targets):
                targets[i, j] = processed_sample.targets[target]
        logger.info("Done!")

        return features, targets
