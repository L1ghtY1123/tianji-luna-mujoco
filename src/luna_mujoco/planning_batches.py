"""Shared candidate batching for the public grasp and held-place planners."""

TRAJECTORY_BATCH_CAPACITY = 16


def trajectory_batches(candidates, capacity=TRAJECTORY_BATCH_CAPACITY):
    """Keep ranked real candidates in bounded, ordered trajectory batches."""
    if type(capacity) is not int or not 1 <= capacity <= TRAJECTORY_BATCH_CAPACITY:
        raise ValueError('invalid trajectory batch capacity')
    return [candidates[start:start + capacity]
            for start in range(0, len(candidates), capacity)]


def pad_trajectory_batch(candidates, capacity=TRAJECTORY_BATCH_CAPACITY):
    """Repeat the final real candidate only to fill the fixed solver capacity."""
    if not 1 <= len(candidates) <= capacity <= TRAJECTORY_BATCH_CAPACITY:
        raise ValueError(f"trajectory batch requires 1 to {capacity} candidates")
    values = tuple(candidates)
    return values + (values[-1],) * (capacity - len(values))
