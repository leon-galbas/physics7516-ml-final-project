from __future__ import annotations

import json
from typing import Iterator

import h5py

from src.data.sample import RawSample


class DataRepository:
    """H5 repository for raw simulated trajectory samples."""

    def __init__(self, path: str, mode: str = "a") -> None:
        """Initializes the repository

        Args:
            path (str): Path to the HDF5 file.
            mode (str, optional): h5py file mode, e.g. 'a', 'r', 'w'. Defaults to "a".
        """
        self.path = path
        self.file = h5py.File(path, mode)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> DataRepository:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        if "t" not in self.file:
            return 0
        else:
            return self.file["t"].shape[0]  # pyright: ignore[reportAttributeAccessIssue]

    def _initialize_from_sample(self, sample: RawSample) -> None:
        string_dtype = h5py.string_dtype(encoding="utf-8")

        self.file.create_dataset(
            "t",
            shape=(0, *sample.t.shape),
            maxshape=(None, *sample.t.shape),
            chunks=(1, *sample.t.shape),
            dtype=sample.t.dtype,
        )

        self.file.create_dataset(
            "position",
            shape=(0, *sample.position.shape),
            maxshape=(None, *sample.position.shape),
            chunks=(1, *sample.position.shape),
            dtype=sample.position.dtype,
        )

        self.file.create_dataset(
            "launch_params",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype=string_dtype,
        )

        self.file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype=string_dtype,
        )

        self.file.attrs["t_shape"] = sample.t.shape
        self.file.attrs["position_shape"] = sample.position.shape

    def append(self, sample: RawSample) -> None:
        if "t" not in self.file:
            self._initialize_from_sample(sample)

        expected_t_shape = tuple(self.file.attrs["t_shape"])  # pyright: ignore[reportArgumentType]
        expected_position_shape = tuple(self.file.attrs["position_shape"])  # pyright: ignore[reportArgumentType]

        if sample.t.shape != expected_t_shape:
            raise ValueError(
                f"Invalid t shape {sample.t.shape}; expected {expected_t_shape}"
            )

        if sample.position.shape != expected_position_shape:
            raise ValueError(
                f"Invalid position shape {sample.position.shape}; "
                f"expected {expected_position_shape}"
            )

        i = len(self)
        new_size = i + 1

        self.file["t"].resize(new_size, axis=0)  # pyright: ignore[reportAttributeAccessIssue]
        self.file["position"].resize(new_size, axis=0)  # pyright: ignore[reportAttributeAccessIssue]
        self.file["launch_params"].resize(new_size, axis=0)  # pyright: ignore[reportAttributeAccessIssue]
        self.file["metadata"].resize(new_size, axis=0)  # pyright: ignore[reportAttributeAccessIssue]

        self.file["t"][i] = sample.t  # pyright: ignore[reportIndexIssue]
        self.file["position"][i] = sample.position  # pyright: ignore[reportIndexIssue]
        self.file["launch_params"][i] = json.dumps(sample.launch_params)  # pyright: ignore[reportIndexIssue]
        self.file["metadata"][i] = json.dumps(sample.metadata)  # pyright: ignore[reportIndexIssue]

        self.file.flush()

    def append_many(self, samples: list[RawSample]) -> None:
        for sample in samples:
            self.append(sample)

    def _read_one(self, index: int) -> RawSample:
        n = len(self)

        if index < 0:
            index += n

        if index < 0 or index >= n:
            raise IndexError(index)

        launch_params_raw = self.file["launch_params"][index]  # pyright: ignore[reportIndexIssue]
        metadata_raw = self.file["metadata"][index]  # pyright: ignore[reportIndexIssue]

        if isinstance(launch_params_raw, bytes):
            launch_params_raw = launch_params_raw.decode("utf-8")

        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")

        return RawSample(
            t=self.file["t"][index],  # pyright: ignore[reportIndexIssue, reportArgumentType]
            position=self.file["position"][index],  # pyright: ignore[reportIndexIssue, reportArgumentType]
            launch_params=json.loads(launch_params_raw),  # pyright: ignore[reportArgumentType]
            metadata=json.loads(metadata_raw),  # pyright: ignore[reportArgumentType]
        )

    def __getitem__(self, index: int | slice) -> RawSample | list[RawSample]:
        if isinstance(index, int):
            return self._read_one(index)

        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self._read_one(i) for i in indices]

        raise TypeError(f"Invalid index type: {type(index)}")

    def __iter__(self) -> Iterator[RawSample]:
        for i in range(len(self)):
            yield self[i]  # pyright: ignore[reportReturnType]
