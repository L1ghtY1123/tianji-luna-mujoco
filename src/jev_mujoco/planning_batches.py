"""Shared candidate batching for the public grasp and held-place planners."""

TRAJECTORY_BATCH_CAPACITY = 16


def trajectory_batches(candidates):
    """Keep ranked real candidates in bounded, ordered trajectory batches."""
    return [candidates[start:start + TRAJECTORY_BATCH_CAPACITY]
            for start in range(0, len(candidates), TRAJECTORY_BATCH_CAPACITY)]


def pad_trajectory_batch(candidates):
    """Repeat the final real candidate only to fill the fixed solver capacity."""
    if not 1 <= len(candidates) <= TRAJECTORY_BATCH_CAPACITY:
        raise ValueError(f"trajectory batch requires 1 to {TRAJECTORY_BATCH_CAPACITY} candidates")
    values = tuple(candidates)
    return values + (values[-1],) * (TRAJECTORY_BATCH_CAPACITY - len(values))
