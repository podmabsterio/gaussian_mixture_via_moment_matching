import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

from time import perf_counter

from threadpoolctl import threadpool_limits

from experiments.utils.dict_utils import data_parallel_metrics_rebuild as rebuild
from experiments.utils.params_shape_utils import convert_and_check_params


def _init_model(
    model_cfg, random_state, data_generator_cfg, init_models_with_oracle_n_components
):
    if init_models_with_oracle_n_components:
        return instantiate(
            model_cfg.target,
            random_state=random_state,
            n_components=data_generator_cfg.n_components,
        )
    else:
        return instantiate(model_cfg.target, random_state=random_state)


def run_on_dataset(
    models_cfg,
    data_generator_cfg,
    evaluator,
    num_model_seeds,
    dataset_seed,
    threads_limit,
    init_models_with_oracle_n_components,
):
    dataset_generator = instantiate(data_generator_cfg)
    dataset = dataset_generator.generate(dataset_seed)
    results = defaultdict(list)
    with threadpool_limits(limits=threads_limit):
        for model_cfg in models_cfg:
            for seed in range(1, num_model_seeds + 1):
                model = _init_model(
                    model_cfg,
                    seed,
                    data_generator_cfg,
                    init_models_with_oracle_n_components,
                )
                try:
                    fit_start = perf_counter()
                    model.fit(**dataset)
                    fit_time = perf_counter() - fit_start

                    model_params = convert_and_check_params(model.params_dict())

                    model_results = evaluator(
                        dataset=dataset, estimated=model_params, model=model
                    )
                    model_results["fit_time"] = fit_time
                    model_results["ok"] = 1
                except:
                    model_results = evaluator.create_empty_metrics() # TODO exclude results with error from aggregation
                    model_results["fit_time"] = 0
                    model_results["ok"] = 0


                results[model_cfg.model_name].append(model_results)

    return int(dataset_seed), results


def raw_results_dataframe(dataset_results):
    """Flatten paired per-run metrics while retaining both random seeds."""
    rows = []
    for dataset_seed, results in dataset_results:
        for model_name, metrics_per_seed in results.items():
            for model_seed, metrics in enumerate(metrics_per_seed, start=1):
                rows.append(
                    {
                        "dataset_seed": int(dataset_seed),
                        "model_seed": int(model_seed),
                        "model_name": model_name,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


class DataParallelRunner:
    def __init__(
        self,
        models_config,
        evaluator,
        threads_limit=1,
        num_datasets=1000,
        model_seeds_per_dataset=1,
        n_jobs=8,
        backend="loky",
        batch_size="auto",
        init_models_with_oracle_n_components=True,
        dataset_seed_start=1,
        save_raw_results=False,
    ):
        self.threads_limit = threads_limit
        self.models_cfg = models_config
        self.num_datasets = num_datasets
        self.model_seeds_per_dataset = model_seeds_per_dataset
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.init_models_with_oracle_n_components = init_models_with_oracle_n_components
        self.dataset_seed_start = int(dataset_seed_start)
        self.save_raw_results = bool(save_raw_results)
        if self.dataset_seed_start < 0:
            raise ValueError("dataset_seed_start must be non-negative")
        self.raw_results_ = None

    def run(self, data_generator_config):
        parallel = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            batch_size=self.batch_size,
            return_as="generator_unordered",
            verbose=0,
        )

        dataset_seeds = list(
            range(
                self.dataset_seed_start,
                self.dataset_seed_start + self.num_datasets,
            )
        )

        results_iter = parallel(
            delayed(run_on_dataset)(
                models_cfg=self.models_cfg,
                data_generator_cfg=data_generator_config,
                evaluator=self.evaluator,
                num_model_seeds=self.model_seeds_per_dataset,
                dataset_seed=dataset_seed,
                threads_limit=self.threads_limit,
                init_models_with_oracle_n_components=self.init_models_with_oracle_n_components,
            )
            for dataset_seed in dataset_seeds
        )

        seeded_results = []

        for dataset_seed, dataset_result in tqdm(
            results_iter,
            total=len(dataset_seeds),
            desc="Datasets",
        ):
            seeded_results.append((dataset_seed, dataset_result))

        if self.save_raw_results:
            self.raw_results_ = raw_results_dataframe(seeded_results)
        else:
            self.raw_results_ = None

        return rebuild([result for _, result in seeded_results])
