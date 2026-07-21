from pathlib import Path

import matplotlib.pyplot as plt
from hydra.utils import instantiate
from sklearn.decomposition import PCA


def plot_data_2d_projection(data_config, save_path, num_samples=3):
    dir_path = Path(save_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    dataset_generator = instantiate(data_config)

    for seed in range(num_samples):
        dataset = dataset_generator.generate(seed + 1)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(dataset["X"])
        means_2d = pca.transform(dataset["true_means"])

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            ax.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=dataset.get("true_labels"),
                s=10,
                alpha=0.7,
                cmap="tab10",
            )
            ax.scatter(
                means_2d[:, 0],
                means_2d[:, 1],
                s=50,
                marker="x",
                c="red",
                linewidths=2,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Sample {seed + 1}")
            fig.tight_layout()
            fig.savefig(dir_path / f"sample_{seed + 1}.png", dpi=150)
        finally:
            plt.close(fig)
