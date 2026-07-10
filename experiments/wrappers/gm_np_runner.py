from experiments.runners.base_runner import BaseRunner


class GmNumpyRunner(BaseRunner):
    def _extract_parameters_not_safe(self):
        params = {}
        if self.use_full_covariance:
            params["covariances"] = self._convert_to_full_covariances(self.gm.sigmas_)
        else:
            params["covariances"] = self.gm.sigmas_

        params["means"] = self.gm.means_
        params["weights"] = self.gm.weights_

        return params
