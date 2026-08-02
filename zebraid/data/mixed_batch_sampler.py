"""
zebraid/data/mixed_batch_sampler.py
⭐ NOVEL — MixedPopulationBatchSampler

Constructs every training batch so that it contains individuals from BOTH
population A and population B.  This is the critical design decision that
forces the embedding space to encode stripe-pattern similarity rather than
population-specific visual artifacts (lighting, background, camera type,
subspecies stripe density).

Without this, a model trained on population A only will learn features that
discriminate within A but may not transfer to B — this is the "generalization
gap" that the ZebraID paper quantifies and closes.

Implementation notes:
  - Each batch contains ceil(batch_size * ratio_a) samples from Pop A
    and floor(batch_size * (1 - ratio_a)) samples from Pop B.
  - Within each population, samples are grouped by individual ID so that
    the triplet loss has valid anchor/positive/negative triplets.
  - The sampler is compatible with torch.utils.data.DataLoader
    (implements __iter__ and __len__).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler

from zebraid.data.dataset import CombinedZebraDataset, POP_A, POP_B


class MixedPopulationBatchSampler(Sampler[list[int]]):
    """
    Yields batches of dataset indices that always contain samples from
    both population A and population B.

    Args:
        dataset:    A CombinedZebraDataset.
        batch_size: Total number of samples per batch.
        ratio_a:    Fraction of each batch drawn from population A (default 0.5).
        drop_last:  If True, drops the last incomplete batch.
        seed:       Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: CombinedZebraDataset,
        batch_size: int,
        ratio_a: float = 0.5,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 < ratio_a < 1.0:
            raise ValueError(f"ratio_a must be in (0, 1), got {ratio_a}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_a = math.ceil(batch_size * ratio_a)
        self.n_b = batch_size - self.n_a
        self.drop_last = drop_last
        self.seed = seed

        # ── Index samples by (population, individual_id) ────────────────────
        self._pop_individual_to_indices: dict[tuple[int, int], list[int]] = defaultdict(
            list
        )
        for ds in dataset.datasets:
            base_offset = dataset._offsets[dataset.datasets.index(ds)]
            for local_idx, sample in enumerate(ds.samples):
                global_idx = base_offset + local_idx
                key = (sample["population_label"], sample["individual_id"])
                self._pop_individual_to_indices[key].append(global_idx)

        # Separate individuals by population
        self._individuals_a: list[tuple[int, int]] = [
            k for k in self._pop_individual_to_indices if k[0] == POP_A
        ]
        self._individuals_b: list[tuple[int, int]] = [
            k for k in self._pop_individual_to_indices if k[0] == POP_B
        ]

        if not self._individuals_a:
            raise ValueError("No population A samples found in dataset.")
        if not self._individuals_b:
            raise ValueError("No population B samples found in dataset.")

        # Estimate number of batches
        total_samples = len(dataset)
        self._num_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size != 0:
            self._num_batches += 1

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)

        def _sample_indices(individuals: list[tuple[int, int]], n: int) -> list[int]:
            """Randomly pick n indices from a pool of individuals."""
            indices: list[int] = []
            shuffled = individuals.copy()
            rng.shuffle(shuffled)
            for ind_key in shuffled:
                pool = self._pop_individual_to_indices[ind_key]
                indices.append(rng.choice(pool))
                if len(indices) >= n:
                    break
            # If we ran out of individuals before reaching n, cycle
            while len(indices) < n:
                ind_key = rng.choice(individuals)
                pool = self._pop_individual_to_indices[ind_key]
                indices.append(rng.choice(pool))
            return indices[:n]

        for _ in range(self._num_batches):
            batch_a = _sample_indices(self._individuals_a, self.n_a)
            batch_b = _sample_indices(self._individuals_b, self.n_b)
            batch = batch_a + batch_b
            rng.shuffle(batch)
            yield batch
