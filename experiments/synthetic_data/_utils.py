import numpy as np


def make_rng(seed, salt):
    """Create a deterministic RNG stream independent of the Gaussian sampler."""
    if seed is None:
        return np.random.default_rng()
    seed_sequence = np.random.SeedSequence([int(seed), int(salt)])
    return np.random.default_rng(seed_sequence)


def contamination_mask(n_samples, contamination_fraction, rng):
    """Select an exact, reproducible number of contaminated observations."""
    fraction = float(contamination_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("contamination_fraction must be finite and in [0, 1)")

    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    n_contaminated = min(int(round(fraction * n_samples)), n_samples - 1)
    mask = np.zeros(n_samples, dtype=bool)
    if n_contaminated:
        indices = rng.choice(n_samples, size=n_contaminated, replace=False)
        mask[indices] = True
    return mask
