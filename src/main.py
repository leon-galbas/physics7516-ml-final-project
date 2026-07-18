import logging

from src.simulation.generator import TrajectoryGenerator
from src.utils import read_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# read config
config = read_config("experiments/test_experiment/config.yaml")

# get simulation config
sim_config = config["simulation"]
n_samples = sim_config["n_samples"]

generator = TrajectoryGenerator(**sim_config)
generator.generate_to_repo(sim_config["repo_name"], n_samples, verbose=True)
