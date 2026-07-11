# physics7516-ml-final-project

Repository for the final project of the 2026 Summer Term Lecture "physics7516:
Machine Learning for Quantum Scientists" at the University of Bonn

## Installation

In order to install and run the code in this package, it is easiest to use the
`uv` package manager. This can for example be installed from the official
homepage via

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once you have `uv` installed, you simply need to clone this repo and inside the
repo root execute

```bash
uv sync
```

## Usage

### Generating data

Individual python scripts can be run via

```bash
uv run src/<script>.py <args>
```

In order to generate a dataset of trajectories you can execute the script
`src/data/make_dataset.py` with a given dataset name. For testing, try

```bash
uv run src/data/make_dataset.py example
```

This will generate an example dataset of 1000 trajectories consisting of 100
timepoints each, as specified in `config/dataset/example.yaml`. It is also
possible to generate data with gaussian or random-walk noise. Example
configurations for that can be found in `example_gauss-noise.yaml` and
`example_random-walk-noise.yaml` and can be run via

```bash
uv run src/data/make_dataset.py example_gauss-noise
uv run src/data/make_dataset.py example_random-walk-noise
```

### Plotting trajectories

Helper functions for plotting trajectories can be found in
`src/visualization/trajectory.py`.

Further Information will follow as the project progresses.
