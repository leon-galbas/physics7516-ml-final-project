import logging

from src.data.repository import DataRepository
from src.data.sample import ProcessedSample
from src.data.transforms.noise import GaussianNoise
from src.data.transforms.sampling import Resample, TimeSlice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# # read config
# config = read_config("experiments/test_experiment/config.yaml")

# # get simulation config
# sim_config = config["simulation"]
# n_samples = sim_config["n_samples"]

# generator = TrajectoryGenerator(**sim_config)
# generator.generate_to_repo(sim_config["repo_name"], n_samples, verbose=True)

with DataRepository("data/raw/10000p_default_ranges.h5", "r") as repo:
    raw_samples = repo[:10]

processed_samples = []

sampler = Resample(500)
slicer = TimeSlice(0, 5)
noise = GaussianNoise(0.1)

for i, sample in enumerate(raw_samples):
    sample = ProcessedSample.from_raw(sample)
    sample = slicer(sample)
    sample = sampler(sample)
    sample = noise(sample)
    processed_samples.append(sample)

print(raw_samples[0])
print(processed_samples[0])
