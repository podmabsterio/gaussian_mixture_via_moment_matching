import torch


class BaseCentersGenerator:
    def __init__(self):
        pass

    def generate(self, X):
        raise NotImplementedError()


class DataCentersGenerator(BaseCentersGenerator):
    def __init__(self):
        super().__init__()

    def generate(self, X):
        return X.clone()


class RandomDataCentersGenerator(BaseCentersGenerator):
    def __init__(self, n_centers=None, noise_std=0.1):
        super().__init__()
        self.n_centers = n_centers
        self.noise_std = noise_std

    def generate(self, X):
        num_samples = X.shape[0]

        if self.n_centers is None:
            n = num_samples
        else:
            n = self.n_centers

        indices = torch.randint(0, num_samples, (n,))
        sampled = X[indices]
        noise = torch.randn_like(sampled) * self.noise_std
        return sampled + noise


class FractionalRandomDataCentersGenerator(BaseCentersGenerator):
    def __init__(self, n_centers_fraction=1.0, noise_std=0.1, min_centers=1):
        super().__init__()
        self.n_centers_fraction = n_centers_fraction
        self.noise_std = noise_std
        self.min_centers = min_centers

    def generate(self, X):
        num_samples = X.shape[0]
        n = max(self.min_centers, int(round(self.n_centers_fraction * num_samples)))

        indices = torch.randint(0, num_samples, (n,), device=X.device)
        sampled = X[indices]
        noise = torch.randn_like(sampled) * self.noise_std
        return sampled + noise
