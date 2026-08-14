"""Dataset tags for the temporary oracle test-center mechanism experiment."""

from src_np.gmm.experimental_oracle_centers import (
    EXPERIMENTAL_CENTER_SCENARIOS,
)

from .sampler_core import UniformBackgroundDatasetGenerator


class OracleCenterScenarioUniformBackgroundDatasetGenerator(
    UniformBackgroundDatasetGenerator
):
    """Attach a center-construction scenario to an otherwise unchanged draw."""

    # TODO(research): This wrapper exists only to route oracle mechanism-test
    # metadata through the generic runner.  Remove it together with the oracle
    # center code, or replace it with a non-oracle experimental abstraction.
    def __init__(
        self,
        experimental_test_center_scenario,
        experimental_bandwidth_reference="test_centers",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if experimental_test_center_scenario not in EXPERIMENTAL_CENTER_SCENARIOS:
            raise ValueError(
                "unknown experimental_test_center_scenario: "
                f"{experimental_test_center_scenario!r}"
            )
        self.experimental_test_center_scenario = (
            experimental_test_center_scenario
        )
        if experimental_bandwidth_reference not in (
            "test_centers",
            "all_observed",
        ):
            raise ValueError(
                "experimental_bandwidth_reference must be 'test_centers' or "
                "'all_observed'"
            )
        self.experimental_bandwidth_reference = experimental_bandwidth_reference

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples=n_samples)
        dataset["experimental_test_center_scenario"] = (
            self.experimental_test_center_scenario
        )
        # TODO(research): This bandwidth-routing tag belongs only to the oracle
        # mechanism control and should disappear with the temporary wrapper.
        dataset["experimental_bandwidth_reference"] = (
            self.experimental_bandwidth_reference
        )
        return dataset
