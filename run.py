import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.utils.io_utils import init_saving, save_dict_to_path
from experiments.metrics.evaluator import Evaluator
from experiments.metrics.metrics_per_model import MetricsPerModelAggregator
from experiments.visualization import plot_data_2d_projection

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="experiments/configs", config_name="train")
def main(config):
    save_path = init_saving(config)

    evaluator = Evaluator(config.metrics)
    aggregator = MetricsPerModelAggregator(config.aggregators)

    for dataset_cfg in config.datasets:
        dataset_dir = save_path / dataset_cfg.name
        if config.save_data_visualizations:
            plot_data_2d_projection(
                dataset_cfg.target,
                dataset_dir / "visualization",
                config.num_visualization_samples,
            )
        runner = instantiate(
            config.runner, models_config=config.models, evaluator=evaluator
        )
        results = runner.run(dataset_cfg.target)
        aggregated_results = aggregator.aggregate(results)
        if runner.raw_results_ is not None:
            aggregated_results["raw"] = runner.raw_results_
        save_dict_to_path(dataset_dir, aggregated_results)


if __name__ == "__main__":
    main()
