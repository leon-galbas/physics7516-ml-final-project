from os import path

import torch

import src.data.transforms as tf
from src.config import SIM_REPO_DIR
from src.data.repository import DataRepository
from src.data.sample import ProcessedSample, RawSample

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
        for tf in transform_config:
            name = list(tf.keys())[0]
            kwargs = tf[name]
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
        self.scalar_features = data_config["scalar_features"]
        self.targets = data_config["targets"]

    def build_dataset(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        processed_samples = []

        with DataRepository(self.repo, "r") as repo:
            raw_samples = repo[:n_samples]

        for sample in raw_samples:
            processed_samples.append(self.pipeline(sample))
