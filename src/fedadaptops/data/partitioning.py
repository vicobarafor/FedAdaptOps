from __future__ import annotations

from collections import Counter

import numpy as np


def dirichlet_partition(
    labels: list[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> dict[int, list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    labels = np.asarray(labels)
    classes = np.unique(labels)

    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)
        client_indices = {client_id: [] for client_id in range(num_clients)}

        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
            splits = np.split(cls_indices, split_points)

            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(v) for v in client_indices.values()]
        if min(sizes) >= min_samples_per_client:
            for indices in client_indices.values():
                rng.shuffle(indices)
            return client_indices

    raise RuntimeError(
        f"Could not create Dirichlet partition with min_samples_per_client={min_samples_per_client}. "
        "Try reducing min_samples_per_client, increasing alpha, or reducing num_clients."
    )


def partition_summary(
    labels: list[int] | np.ndarray, partitions: dict[int, list[int]]
) -> list[dict]:
    labels = np.asarray(labels)
    rows = []
    for client_id, indices in partitions.items():
        counts = Counter(labels[indices].tolist())
        rows.append(
            {
                "client_id": client_id,
                "num_samples": len(indices),
                "class_counts": dict(sorted(counts.items())),
            }
        )
    return rows
