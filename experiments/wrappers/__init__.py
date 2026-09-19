from experiments.wrappers.population_gm import PopulationWrapper
from experiments.wrappers.oracle_init_gm import OracleInitWrapper
from experiments.wrappers.sklearn_models import (
    SklearnBayesianGaussianMixtureWrapper,
    SklearnGaussianMixtureWrapper,
    SklearnKMeansWrapper,
)

try:
    from experiments.wrappers.torch_dimension_free_linear import (
        TorchDimensionFreeLinearWrapper,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
