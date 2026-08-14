import numpy as np


def generate_weights(
    n_components,
    concentration=1.0,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    alpha = np.full(n_components, concentration)
    return rng.dirichlet(alpha)
