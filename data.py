import math
from typing import Dict, Callable, Tuple

import numpy as np
import torch

TensorDict = Dict[str, torch.Tensor]

NoiseSampler = Callable[[np.ndarray], np.ndarray]


def _generate_inverse_core(n: int, noise_sampler: NoiseSampler) -> Tuple[np.ndarray, np.ndarray]:
    """Sample y ~ U[0,1], add noise to x = y + 0.3 sin(2πy)."""
    y = np.random.rand(n, 1)
    x_clean = y + 0.3 * np.sin(2 * np.pi * y)
    x = x_clean + noise_sampler(y)
    return y, x


def _pack_xy(y_train: np.ndarray, x_train: np.ndarray, y_test: np.ndarray, x_test: np.ndarray) -> TensorDict:
    return {
        "y_train": torch.from_numpy(y_train).float(),
        "x_train": torch.from_numpy(x_train).float(),
        "y_test": torch.from_numpy(y_test).float(),
        "x_test": torch.from_numpy(x_test).float(),
    }


def generate_inverse_problem_data(
        n_train: int = 1000,
        n_test: int = 1000,
        hetero: bool = False,
        noise_base: float = 0.1,
) -> TensorDict:
    """Inverse toy with optional heteroscedastic noise.

    hetero=False: epsilon ~ U(-0.1, 0.1)
    hetero=True:  epsilon ~ N(0, sigma(y)), sigma(y)=noise_base*(0.1+|y-0.5|)
    """

    if hetero:
        noise = lambda y: np.random.normal(0.0, noise_base * (0.1 + np.abs(y - 0.5)), size=y.shape)
    else:
        noise = lambda y: np.random.uniform(-0.1, 0.1, size=y.shape)

    y_train, x_train = _generate_inverse_core(n_train, noise)
    y_test, x_test = _generate_inverse_core(n_test, noise)
    return _pack_xy(y_train, x_train, y_test, x_test)


def generate_inverse_problem_swapped_data(
        n_train: int = 1000,
        n_test: int = 1000,
        hetero: bool = False,
        noise_base: float = 0.1,
) -> TensorDict:
    """Return inverse splits (input=x, target=y) with uniform or heteroscedastic noise."""

    data = generate_inverse_problem_data(n_train, n_test, hetero=hetero, noise_base=noise_base)

    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
    }


# Generate synthetic dataset 3
def generate_multi_branch_data(n_train: int = 5000, n_test: int = 2000) -> TensorDict:
    """Synthetic multi-branch dataset with input-dependent mixture weights and noise."""

    def sample_set(n: int):
        x = np.random.rand(n, 1)  # x ~ U[0,1]

        w1 = np.exp(-((x - 0.2) / 0.08) ** 2)
        w2 = np.exp(-((x - 0.5) / 0.08) ** 2)
        w3 = np.exp(-((x - 0.8) / 0.08) ** 2)
        W = np.concatenate([w1, w2, w3], axis=1)
        W = W + 1e-6
        W = W / W.sum(axis=1, keepdims=True)

        z = np.array([np.random.choice(3, p=W[i]) for i in range(n)], dtype=int)
        f1 = lambda x: 0.2 + 0.4 * np.sin(2 * np.pi * x)
        f2 = lambda x: 0.7 - 0.3 * np.sin(4 * np.pi * x + 0.3)
        f3 = lambda x: 0.1 + 0.8 * x

        s1 = lambda x: 0.03 + 0.02 * x
        s2 = lambda x: 0.05 + 0.25 * np.exp(-((x - 0.5) / 0.15) ** 2)
        s3 = lambda x: 0.02 + 0.05 * x
        # s4 = lambda x: 0.04 + 0.08 * np.exp(-((x - 0.35) / 0.08) ** 2)

        y = np.zeros_like(x)
        for i in range(n):
            xi = x[i, 0]
            if z[i] == 0:
                mean = f1(xi)
                std = s1(xi)
            elif z[i] == 1:
                mean = f2(xi)
                std = s2(xi)
            elif z[i] == 2:
                mean = f3(xi)
                std = s3(xi)
            # else:
            #    mean = f4(xi)
            #    std = s4(xi)
            y[i, 0] = mean + std * np.random.randn()

        return x, y

    x_train, y_train = sample_set(n_train)
    x_test, y_test = sample_set(n_test)

    return {
        "x_train": torch.from_numpy(x_train).float(),
        "y_train": torch.from_numpy(y_train).float(),
        "x_test": torch.from_numpy(x_test).float(),
        "y_test": torch.from_numpy(y_test).float(),
    }


# inject outliers into multi-branch dataset
def inject_outliers(data_mb, frac: float = 0.0005, low: float = -1.0, high: float = 1.5):
    """Inject a small fraction of outliers into multi-branch training targets."""
    data_noisy = {
        "x_train": data_mb["x_train"].clone(),
        "y_train": data_mb["y_train"].clone(),
        "x_test": data_mb["x_test"],
        "y_test": data_mb["y_test"],
    }

    n_train = data_mb["y_train"].shape[0]
    n_outliers = max(1, int(frac * n_train))
    idx = np.random.choice(n_train, size=n_outliers, replace=False)
    y_rand = torch.empty((n_outliers, 1)).uniform_(low, high)
    data_noisy["y_train"][idx] = y_rand

    outlier_mask = torch.zeros(n_train, dtype=torch.bool)
    outlier_mask[idx] = True

    return data_noisy, outlier_mask


def _sample_polygon_edges(vertices, n_points, jitter: float = 0.0) -> np.ndarray:
    """Sample points uniformly along polygon edges (closed), optional radial jitter."""
    verts = np.array(vertices, dtype=float)
    edges = np.roll(verts, -1, axis=0) - verts
    edge_lengths = np.linalg.norm(edges, axis=1)
    probs = edge_lengths / edge_lengths.sum()
    counts = np.random.multinomial(n_points, probs)
    pts = []
    for v0, v1, cnt in zip(verts, np.roll(verts, -1, axis=0), counts):
        if cnt == 0:
            continue
        t = np.random.rand(cnt, 1)
        p = v0 + t * (v1 - v0)
        if jitter > 0:
            p += np.random.randn(*p.shape) * jitter
        pts.append(p)
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def generate_star_with_circle(
        n_star: int = 1200,
        n_circle: int = 800,
        r_star: float = 1.0,
        r_circle: float = 1.35,
        jitter: float = 0.002,
        train_ratio: float = None,
        as_tensor_dataset: bool = False,
):
    """Generate a hexagram (two triangles) plus an outer circle of points.

    If train_ratio is set (0<train_ratio<1), returns a split (train, test).
    If as_tensor_dataset is True, returns TensorDataset; otherwise tensors.
    """
    angles1 = np.deg2rad([90, 210, 330])
    angles2 = angles1 + np.deg2rad(60)
    tri1 = np.stack([r_star * np.cos(angles1), r_star * np.sin(angles1)], axis=1)
    tri2 = np.stack([r_star * np.cos(angles2), r_star * np.sin(angles2)], axis=1)

    star_pts = np.concatenate(
        [
            _sample_polygon_edges(tri1, n_star // 2, jitter=jitter),
            _sample_polygon_edges(tri2, n_star // 2, jitter=jitter),
        ],
        axis=0,
    )

    angles = np.random.rand(n_circle) * 2 * math.pi
    x_circle = r_circle * np.cos(angles)
    y_circle = r_circle * np.sin(angles)
    circle_pts = np.stack([x_circle, y_circle], axis=1)
    if jitter > 0:
        circle_pts += np.random.randn(*circle_pts.shape) * jitter

    data = np.concatenate([star_pts, circle_pts], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(star_pts), dtype=int),
            np.ones(len(circle_pts), dtype=int),
        ]
    )

    data_t = torch.from_numpy(data).float()
    labels_t = torch.from_numpy(labels)

    if train_ratio is None:
        return {"data": data_t, "labels": labels_t}

    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0,1)"
    n_total = data_t.shape[0]
    n_train = int(train_ratio * n_total)
    idx = torch.randperm(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    x_train, y_train = data_t[train_idx], labels_t[train_idx]
    x_test, y_test = data_t[test_idx], labels_t[test_idx]

    if as_tensor_dataset:
        from torch.utils.data import TensorDataset

        return (
            TensorDataset(x_train, y_train),
            TensorDataset(x_test, y_test),
        )
    return x_train, y_train, x_test, y_test


__all__ = [
    "generate_inverse_problem_data",
    "generate_inverse_problem_swapped_data",
    "generate_multi_branch_data",
    "inject_outliers",
    "generate_star_with_circle",
    "TensorDict",
]
