import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.utils.io_utils import init_saving, save_dict_to_path
from experiments.metrics.evaluator import Evaluator
from experiments.metrics.metrics_per_model import MetricsPerModelAggregator

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="experiments/configs", config_name="train")
def main(config):
    save_path = init_saving(config)

    evaluator = Evaluator(config.metrics)
    aggregator = MetricsPerModelAggregator(config.aggregators)

    for dataset_cfg in config.datasets:
        runner = instantiate(
            config.runner, models_config=config.models, evaluator=evaluator
        )
        results = runner.run(dataset_cfg.target)
        aggregated_results = aggregator.aggregate(results)
        save_dict_to_path(save_path / dataset_cfg.name, aggregated_results)


if __name__ == "__main__":
    main()
