class BaseInitializer:
    def __init__(self, n_components):
        self.n_components = n_components

    def get_initial_params(self, X):
        raise NotImplementedError()
