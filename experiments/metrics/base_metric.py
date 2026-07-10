class BaseMetric:
    def __init__(self, name):
        self.name = name  # all subclasses should pass short but readable metric name in super().__init__()

    def __call__(self, *args, **kwds):
        """
        All subclasses can have any arguments they need for evaluation and then add **kwargs for unified API.
        Possible kwargs:
        covariances - estimated covariances
        means - estimated means
        weights - estimated weights
        labels - estimated labels
        true_covariances - true covariances
        true_means - true means
        true_weights - true weights
        true_labels - true labels
        X - training data
        For generating new samples metrics should use function sample_gm_data(means, weights, covariances, num_samples, seed=1)
        """
        raise NotImplementedError()
