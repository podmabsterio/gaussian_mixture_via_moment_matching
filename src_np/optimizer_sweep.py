"""Config-driven comparison of NumPy one-s GMM fits and the Torch baseline.

Run from the repository root:

    python -m src_np.optimizer_sweep \
        --config gmm_numpy/optimizer_sweep.example.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.gmm import OneSGaussianMixtureModel
from src_np.kmeans_initialization import initialize_dimension_free_one_s_gmm
from src_np.metrics import gmm_test_log_likelihood, matched_center_distance
from src.data_generation.core import MixtureDatasetGenerator
from src.data_generation.covarience_generator import IsotropicCovarianceSampler
from src.data_generation.dataclass import MixtureSpec
from src.data_generation.means_generator import FixedDistanceCenterSampler
from src.data_generation.weight_generator import (
    DirichletWeightSampler,
    EqualWeightSampler,
)


RAW_FIELDS = [
    "dataset_id",
    "scenario",
    "seed",
    "n_samples",
    "n_features",
    "n_components",
    "separation",
    "method",
    "method_family",
    "num_directions",
    "joint_optimization",
    "include_base_kernel",
    "s",
    "target_neighbor_count",
    "fit_time_sec",
    "matched_center_distance",
    "test_log_likelihood",
    "sigma_mae",
    "weight_l1",
    "objective",
    "n_iter",
    "status",
    "error",
]

SUMMARY_METRICS = [
    "fit_time_sec",
    "matched_center_distance",
    "test_log_likelihood",
    "sigma_mae",
    "weight_l1",
    "objective",
    "n_iter",
]

DEFAULT_COMPARISON_METHODS = [
    "kmeans",
    "gmm_torch_lbfgs",
    "numpy_componentwise_r3",
]

COMPARISON_LABELS = {
    "kmeans": "KMeans",
    "gmm_torch_lbfgs": "Torch",
    "numpy_componentwise_r3": "NumPy R3",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _as_list(value):
    return value if isinstance(value, list) else [value]


def expand_dataset_grid(config):
    defaults = config.get("data_defaults", {})
    for scenario in config["scenarios"]:
        scenario_name = scenario["name"]
        n_values = _as_list(scenario["n_samples"])
        d_values = _as_list(scenario.get("n_features", defaults.get("n_features")))
        k_values = _as_list(scenario["n_components"])
        for n_samples, n_features, n_components in itertools.product(
            n_values, d_values, k_values
        ):
            parameters = dict(defaults)
            parameters.update(
                {
                    key: value
                    for key, value in scenario.items()
                    if key not in {"name", "n_samples", "n_features", "n_components"}
                }
            )
            parameters.update(
                n_samples=int(n_samples),
                n_features=int(n_features),
                n_components=int(n_components),
                scenario=scenario_name,
            )
            if parameters["n_samples"] < parameters["n_components"]:
                raise ValueError(
                    f"{scenario_name}: n_samples must be at least n_components"
                )
            if parameters["n_components"] > parameters["n_features"] + 1:
                raise ValueError(
                    f"{scenario_name}: FixedDistanceCenterSampler requires K <= d + 1"
                )
            yield parameters


def generate_dataset(parameters, seed):
    weight_mode = parameters.get("weight_mode", "uniform")
    if weight_mode == "uniform":
        weight_sampler = EqualWeightSampler()
    elif weight_mode == "dirichlet":
        weight_sampler = DirichletWeightSampler(
            concentration=float(parameters.get("weight_concentration", 2.0))
        )
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")

    generator = MixtureDatasetGenerator(
        weight_sampler=weight_sampler,
        center_sampler=FixedDistanceCenterSampler(),
        covariance_sampler=IsotropicCovarianceSampler(),
    )
    spec = MixtureSpec(
        n_samples=parameters["n_samples"],
        n_features=parameters["n_features"],
        n_components=parameters["n_components"],
        separation=float(parameters.get("separation", 3.0)),
        variance_spread=float(parameters.get("variance_spread", 0.25)),
        weight_mode=weight_mode,
        weight_concentration=float(parameters.get("weight_concentration", 2.0)),
        base_scale=float(parameters.get("base_scale", 1.0)),
        random_seed=int(seed),
        scenario=parameters["scenario"],
    )
    return generator.generate(spec)


def select_single_bandwidth(X, n_components, bandwidth_config):
    if "target_neighbor_count" in bandwidth_config:
        target = float(bandwidth_config["target_neighbor_count"])
    else:
        fraction = float(bandwidth_config.get("neighbors_per_component_fraction", 0.5))
        target = fraction * X.shape[0] / n_components
    target = min(max(target, 1.0 + 1e-6), X.shape[0] * (1.0 - 1e-6))
    s = select_s_values_by_average_kernel_count(
        X,
        n_components=n_components,
        target_neighbor_counts=[target],
        decreasing=False,
        count_rtol=float(bandwidth_config.get("count_rtol", 1e-3)),
        s_rtol=float(bandwidth_config.get("s_rtol", 1e-4)),
    )[0]
    return float(s), float(target)


def true_isotropic_sigmas(dataset):
    diagonal = np.diagonal(dataset.covariances, axis1=1, axis2=2)
    return np.sqrt(np.mean(diagonal, axis=1))


def evaluate_parameters(means, sigmas, weights, dataset, evaluation_config, seed):
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / weights.sum()
    true_sigmas = true_isotropic_sigmas(dataset)

    center_distance, assignment = matched_center_distance(
        means, dataset.means, return_assignment=True
    )
    sigma_mae = np.mean(np.abs(sigmas - true_sigmas[assignment]))
    weight_l1 = np.sum(np.abs(weights - dataset.weights[assignment]))
    test_log_likelihood = gmm_test_log_likelihood(
        means,
        sigmas,
        weights,
        dataset.means,
        true_sigmas,
        dataset.weights,
        n_test=int(evaluation_config.get("n_test", 20_000)),
        batch_size=int(evaluation_config.get("batch_size", 8192)),
        random_state=int(seed),
    )
    return {
        "matched_center_distance": float(center_distance),
        "test_log_likelihood": float(test_log_likelihood),
        "sigma_mae": float(sigma_mae),
        "weight_l1": float(weight_l1),
    }


def base_result_row(parameters, seed, dataset_id, s, target):
    return {
        "dataset_id": dataset_id,
        "scenario": parameters["scenario"],
        "seed": seed,
        "n_samples": parameters["n_samples"],
        "n_features": parameters["n_features"],
        "n_components": parameters["n_components"],
        "separation": parameters.get("separation", 3.0),
        "s": s,
        "target_neighbor_count": target,
        "status": "ok",
        "error": "",
    }


def run_kmeans_baseline(dataset, parameters, seed, common_row, evaluation_config):
    start = time.perf_counter()
    result = initialize_dimension_free_one_s_gmm(
        dataset.X,
        n_components=parameters["n_components"],
        s=common_row["s"],
        n_init=int(parameters.get("kmeans_n_init", 20)),
        random_state=int(seed),
    )
    elapsed = time.perf_counter() - start
    row = dict(common_row)
    row.update(
        method="kmeans",
        method_family="baseline",
        num_directions=0,
        joint_optimization=False,
        include_base_kernel=False,
        fit_time_sec=elapsed,
        objective=np.nan,
        n_iter=0,
    )
    row.update(
        evaluate_parameters(
            result["means"],
            result["sigmas"],
            result["weights"],
            dataset,
            evaluation_config,
            seed=1_000_000 + int(seed),
        )
    )
    return row


def run_numpy_model(
    dataset,
    parameters,
    seed,
    common_row,
    evaluation_config,
    model_config,
    num_directions,
    joint_optimization,
):
    model = OneSGaussianMixtureModel(
        k=parameters["n_components"],
        s_values=[common_row["s"]],
        n_init=int(model_config.get("n_init", 20)),
        mean_penalty_weight=float(model_config.get("mean_penalty_weight", 0.0)),
        objective_rtol=float(model_config.get("objective_rtol", 1e-6)),
        objective_atol=float(model_config.get("objective_atol", 0.0)),
        convergence_patience=int(model_config.get("convergence_patience", 2)),
        num_directions=int(num_directions),
        joint_optimization=bool(joint_optimization),
        random_state=int(seed),
    )
    start = time.perf_counter()
    model.fit(dataset.X)
    elapsed = time.perf_counter() - start
    fit_result = model.history_per_s[common_row["s"]]
    optimization_name = "joint" if joint_optimization else "componentwise"
    row = dict(common_row)
    row.update(
        method=f"numpy_{optimization_name}_r{num_directions}",
        method_family="numpy_one_s",
        num_directions=num_directions,
        joint_optimization=joint_optimization,
        include_base_kernel=False,
        fit_time_sec=elapsed,
        objective=float(fit_result["objective"]),
        n_iter=int(fit_result["n_iter"]),
    )
    row.update(
        evaluate_parameters(
            model.means_,
            model.sigmas_,
            model.weights_,
            dataset,
            evaluation_config,
            seed=1_000_000 + int(seed),
        )
    )
    return row


def run_torch_model(
    dataset,
    parameters,
    seed,
    common_row,
    evaluation_config,
    model_config,
):
    import torch

    from src.gmm import (
        FixedBandwidthGenerator,
        FullBatchStrategy,
        GaussianMomentModelDimensionFreeLinear,
        KMeansInitializer,
        Trainer,
    )

    torch.manual_seed(int(seed))
    dtype_name = model_config.get("dtype", "float64")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    device = torch.device(model_config.get("device", "cpu"))
    X = torch.as_tensor(dataset.X, dtype=dtype, device=device)
    s = common_row["s"]
    initializer = KMeansInitializer(
        n_components=parameters["n_components"],
        n_init=int(model_config.get("n_init", 20)),
        random_state=int(seed),
    )
    bandwidth_generator = FixedBandwidthGenerator(1, s, s)
    model = GaussianMomentModelDimensionFreeLinear(
        initializer=initializer,
        bandwidth_generator=bandwidth_generator,
        n_components=parameters["n_components"],
        num_directions=int(model_config.get("num_directions", 3)),
        include_base_kernel=bool(model_config.get("include_base_kernel", False)),
    )
    optimizer_factory = partial(
        torch.optim.LBFGS,
        max_iter=int(model_config.get("lbfgs_max_iter", 20)),
        tolerance_grad=float(model_config.get("tolerance_grad", 1e-7)),
        tolerance_change=float(model_config.get("tolerance_change", 1e-9)),
        history_size=int(model_config.get("history_size", 100)),
        line_search_fn=model_config.get("line_search_fn", None),
    )
    strategy = FullBatchStrategy(
        optimizer_factory=optimizer_factory,
        n_steps=int(model_config.get("outer_steps", 10)),
    )
    trainer = Trainer(model=model, strategy=strategy)
    start = time.perf_counter()
    trainer.fit(X)
    elapsed = time.perf_counter() - start

    means = trainer.means_.detach().cpu().numpy()
    sigmas = np.sqrt(trainer.covariances_.detach().cpu().numpy())
    weights = trainer.weights_.detach().cpu().numpy()
    row = dict(common_row)
    row.update(
        method="gmm_torch_lbfgs",
        method_family="torch",
        num_directions=int(model_config.get("num_directions", 3)),
        joint_optimization=True,
        include_base_kernel=bool(model_config.get("include_base_kernel", False)),
        fit_time_sec=elapsed,
        objective=float(model.loss().detach().cpu().item()),
        n_iter=int(model_config.get("outer_steps", 10)),
    )
    row.update(
        evaluate_parameters(
            means,
            sigmas,
            weights,
            dataset,
            evaluation_config,
            seed=1_000_000 + int(seed),
        )
    )
    return row


def failure_row(common_row, method, family, error, **method_fields):
    row = dict(common_row)
    row.update(
        method=method,
        method_family=family,
        status="error",
        error=f"{type(error).__name__}: {error}",
        fit_time_sec=np.nan,
        matched_center_distance=np.nan,
        test_log_likelihood=np.nan,
        sigma_mae=np.nan,
        weight_l1=np.nan,
        objective=np.nan,
        n_iter=np.nan,
        include_base_kernel=False,
    )
    row.update(method_fields)
    return row


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    group_fields = [
        "scenario",
        "n_samples",
        "n_features",
        "n_components",
        "method",
        "method_family",
        "num_directions",
        "joint_optimization",
        "include_base_kernel",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)

    summary = []
    for key, group in grouped.items():
        output = dict(zip(group_fields, key))
        successful = [row for row in group if row["status"] == "ok"]
        output["n_success"] = len(successful)
        output["n_failed"] = len(group) - len(successful)
        for metric in SUMMARY_METRICS:
            values = np.asarray([row[metric] for row in successful], dtype=float)
            output[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            output[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0 if values.size else np.nan
            )
        summary.append(output)
    summary.sort(key=lambda row: tuple(str(row.get(field)) for field in group_fields))
    return summary


def build_compact_comparison(summary, methods=None):
    methods = methods or DEFAULT_COMPARISON_METHODS
    group_fields = ["scenario", "n_samples", "n_features", "n_components"]
    grouped = defaultdict(dict)
    for row in summary:
        if row["method"] in methods:
            key = tuple(row[field] for field in group_fields)
            grouped[key][row["method"]] = row

    compact_rows = []
    for key, method_rows in grouped.items():
        if not all(method in method_rows for method in methods):
            continue
        output = dict(zip(group_fields, key))
        output["n_seeds"] = min(
            int(method_rows[method]["n_success"]) for method in methods
        )
        for method in methods:
            source = method_rows[method]
            for metric in (
                "matched_center_distance",
                "test_log_likelihood",
                "sigma_mae",
                "weight_l1",
                "fit_time_sec",
            ):
                output[f"{method}_{metric}"] = source[f"{metric}_mean"]

        minimization_metrics = {
            "best_center_method": "matched_center_distance",
            "best_sigma_method": "sigma_mae",
            "best_weight_method": "weight_l1",
            "fastest_method": "fit_time_sec",
        }
        for output_field, metric in minimization_metrics.items():
            winner = min(
                methods, key=lambda method: method_rows[method][f"{metric}_mean"]
            )
            output[output_field] = COMPARISON_LABELS.get(winner, winner)
        tll_winner = max(
            methods,
            key=lambda method: method_rows[method]["test_log_likelihood_mean"],
        )
        output["best_tll_method"] = COMPARISON_LABELS.get(tll_winner, tll_winner)
        compact_rows.append(output)

    compact_rows.sort(
        key=lambda row: (
            str(row["scenario"]),
            int(row["n_components"]),
            -int(row["n_samples"]),
        )
    )
    return compact_rows


def write_compact_markdown(path, rows, methods=None):
    methods = methods or DEFAULT_COMPARISON_METHODS
    headers = ["Scenario", "n", "K"]
    for method in methods:
        label = COMPARISON_LABELS.get(method, method)
        headers.extend([f"{label} center", f"{label} TLL", f"{label} sec"])
    headers.extend(["Best center", "Best TLL"])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row["scenario"]), str(row["n_samples"]), str(row["n_components"])]
        for method in methods:
            values.extend(
                [
                    f"{float(row[f'{method}_matched_center_distance']):.4f}",
                    f"{float(row[f'{method}_test_log_likelihood']):.4f}",
                    f"{float(row[f'{method}_fit_time_sec']):.3f}",
                ]
            )
        values.extend([str(row["best_center_method"]), str(row["best_tll_method"])])
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    raw_rows = []
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary.csv"
    evaluation_config = config.get("evaluation", {})
    numpy_config = config.get("numpy_model", {})
    torch_config = config.get("torch_model", {})
    directions = [int(value) for value in numpy_config.get("num_directions", [1, 2, 3])]

    for parameters in expand_dataset_grid(config):
        for seed in config.get("seeds", [0]):
            dataset_id = (
                f"{parameters['scenario']}_n{parameters['n_samples']}_"
                f"d{parameters['n_features']}_k{parameters['n_components']}_seed{seed}"
            )
            print(f"Running {dataset_id}", flush=True)
            dataset = generate_dataset(parameters, seed)
            s, target = select_single_bandwidth(
                dataset.X,
                parameters["n_components"],
                config.get("bandwidth", {}),
            )
            common_row = base_result_row(parameters, seed, dataset_id, s, target)

            if config.get("include_kmeans_baseline", True):
                try:
                    raw_rows.append(
                        run_kmeans_baseline(
                            dataset, parameters, seed, common_row, evaluation_config
                        )
                    )
                except Exception as error:  # keep long sweeps alive and record failures
                    raw_rows.append(
                        failure_row(
                            common_row,
                            "kmeans",
                            "baseline",
                            error,
                            num_directions=0,
                            joint_optimization=False,
                        )
                    )

            for joint_optimization in (False, True):
                for num_directions in directions:
                    method = (
                        f"numpy_{'joint' if joint_optimization else 'componentwise'}_"
                        f"r{num_directions}"
                    )
                    try:
                        row = run_numpy_model(
                            dataset,
                            parameters,
                            seed,
                            common_row,
                            evaluation_config,
                            numpy_config,
                            num_directions,
                            joint_optimization,
                        )
                    except (
                        Exception
                    ) as error:  # keep long sweeps alive and record failures
                        row = failure_row(
                            common_row,
                            method,
                            "numpy_one_s",
                            error,
                            num_directions=num_directions,
                            joint_optimization=joint_optimization,
                        )
                    raw_rows.append(row)

            if torch_config.get("enabled", True):
                try:
                    raw_rows.append(
                        run_torch_model(
                            dataset,
                            parameters,
                            seed,
                            common_row,
                            evaluation_config,
                            torch_config,
                        )
                    )
                except Exception as error:  # keep long sweeps alive and record failures
                    raw_rows.append(
                        failure_row(
                            common_row,
                            "gmm_torch_lbfgs",
                            "torch",
                            error,
                            num_directions=int(torch_config.get("num_directions", 3)),
                            joint_optimization=True,
                            include_base_kernel=bool(
                                torch_config.get("include_base_kernel", False)
                            ),
                        )
                    )

            write_csv(raw_path, raw_rows, RAW_FIELDS)
            summary = build_summary(raw_rows)
            summary_fields = list(summary[0]) if summary else []
            if summary_fields:
                write_csv(summary_path, summary, summary_fields)
                comparison_methods = config.get(
                    "comparison_methods", DEFAULT_COMPARISON_METHODS
                )
                compact = build_compact_comparison(summary, comparison_methods)
                if compact:
                    write_csv(
                        output_dir / "best_methods.csv",
                        compact,
                        list(compact[0]),
                    )
                    write_compact_markdown(
                        output_dir / "best_methods.md",
                        compact,
                        comparison_methods,
                    )

    return raw_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the JSON config")
    args = parser.parse_args()
    run_sweep(load_config(args.config))


if __name__ == "__main__":
    main()
