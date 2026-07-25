from src.data.sample import ProcessedSample


class ComputeHitpoint:
    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        sample.targets["hit_x"] = sample.trajectory["x"][-1]
        sample.targets["hit_y"] = sample.trajectory["y"][-1]

        return sample
