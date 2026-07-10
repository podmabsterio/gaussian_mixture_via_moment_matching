import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

from time import perf_counter

from threadpoolctl import threadpool_limits

from experiments.utils.dict_utils import data_parallel_metrics_rebuild as rebuild
from experiments.utils.params_shape_utils import convert_and_check_params


def run_on_dataset(
    models_cfg,
    data_generator_cfg,
    evaluator,
    num_model_seeds,
    dataset_seed,
    threads_limit,
):
    dataset_generator = instantiate(data_generator_cfg)
    dataset = dataset_generator.generate(dataset_seed)
    results = defaultdict(list)
    with threadpool_limits(limits=threads_limit):
        for model_cfg in models_cfg:
            for seed in range(1, num_model_seeds + 1):
                model = instantiate(
                    model_cfg.target, random_state=seed, k=dataset.n_components
                )
                fit_start = perf_counter()
                model.fit(**dataset)
                fit_time = perf_counter() - fit_start

                model_params = convert_and_check_params(model.params_dict())

                model_results = evaluator(dataset=dataset, estimated=model_params)
                model_results["fit_time"] = fit_time
                results[model_cfg.model_name].append(model_results)

    return results


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
    ):
        self.threads_limit = threads_limit
        self.models_cfg = models_config
        self.num_datasets = num_datasets
        self.model_seeds_per_dataset = model_seeds_per_dataset
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size

    def run(self, data_generator_config):
        parallel = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            batch_size=self.batch_size,
            return_as="generator_unordered",
            verbose=0,
        )

        dataset_seeds = list(range(1, self.num_datasets + 1))

        results_iter = parallel(
            delayed(run_on_dataset)(
                models_cfg=self.models_cfg,
                data_generator_cfg=data_generator_config,
                evaluator=self.evaluator,
                num_model_seeds=self.model_seeds_per_dataset,
                dataset_seed=dataset_seed,
                threads_limit=self.threads_limit,
            )
            for dataset_seed in dataset_seeds
        )

        nested_results = []

        for dataset_result in tqdm(
            results_iter,
            total=len(dataset_seeds),
            desc="Datasets",
        ):
            nested_results.append(dataset_result)

        return rebuild(nested_results)
