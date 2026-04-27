import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


PARAMETRIZATIONS = ["DDPM", "EDM", "FM", "EqM"]
TARGET_COEFFS = {
    "DDPM": lambda t: (np.zeros_like(t), np.ones_like(t)),
    "EDM": lambda t: (np.ones_like(t), np.zeros_like(t)),
    "FM": lambda t: (-np.ones_like(t), np.ones_like(t)),
    "EqM": lambda t: (-t, t),
}
CMAPS = {
    "DDPM": "Reds",
    "EDM": "Greens",
    "FM": "Blues",
    "EqM": "Purples",
}


rng = np.random.default_rng(89)


def make_swiss_roll(n, jitter):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 12.0
    pts += jitter * rng.standard_normal(pts.shape)
    return pts


def init_mlp(in_dim=2, hidden=64, out_dim=2, seed=0):
    g = np.random.default_rng(seed)
    return dict(
        W1=g.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
        b1=np.zeros(hidden),
        W2=g.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden),
        b2=np.zeros(hidden),
        W3=g.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden),
        b3=np.zeros(out_dim),
    )


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_deriv(x):
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x**3)
    t = np.tanh(u)
    du_dx = c * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du_dx


def forward(p, u):
    h1 = u @ p["W1"] + p["b1"]
    a1 = gelu(h1)
    h2 = a1 @ p["W2"] + p["b2"]
    a2 = gelu(h2)
    out = a2 @ p["W3"] + p["b3"]
    return out, (u, h1, a1, h2, a2)


def backward(p, cache, dout):
    u, h1, a1, h2, a2 = cache
    n = u.shape[0]
    dW3 = a2.T @ dout / n
    db3 = dout.mean(0)
    da2 = dout @ p["W3"].T
    dh2 = da2 * gelu_deriv(h2)
    dW2 = a1.T @ dh2 / n
    db2 = dh2.mean(0)
    da1 = dh2 @ p["W2"].T
    dh1 = da1 * gelu_deriv(h1)
    dW1 = u.T @ dh1 / n
    db1 = dh1.mean(0)
    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)


def sgd_step(p, grads, lr):
    return {k: p[k] - lr * grads[k] for k in p}


def schedule(t):
    return 1.0 - t, t


def train_one(name, X, steps=2500, batch=256, lr=2e-3, seed=0):
    p = init_mlp(hidden=64, seed=seed)
    n = X.shape[0]
    coeff_fn = TARGET_COEFFS[name]
    for _ in range(steps):
        idx = rng.integers(0, n, size=batch)
        xb = X[idx]
        t = rng.random(batch).reshape(-1, 1)
        eps = rng.standard_normal(xb.shape)
        a, b = schedule(t)
        ub = a * xb + b * eps
        c_t, d_t = coeff_fn(t)
        r = c_t * xb + d_t * eps
        pred, cache = forward(p, ub)
        err = pred - r
        dout = 2.0 * err / pred.shape[1]
        grads = backward(p, cache, dout)
        p = sgd_step(p, grads, lr)
    return p


def step_dir(name, pred, u):
    if name == "DDPM":
        return -pred
    if name == "EDM":
        return pred - u
    if name == "FM":
        return -pred
    if name == "EqM":
        return -pred
    raise ValueError(name)


def sample_trajectories(name, model, X, n=200, K=100, dt=0.04, seed=42):
    g = np.random.default_rng(seed)
    u = g.standard_normal((n, 2))
    traj = [u.copy()]
    for _ in range(K):
        pred, _ = forward(model, u)
        u = u + dt * step_dir(name, pred, u)
        u = np.clip(u, -10, 10)
        traj.append(u.copy())
    traj = np.array(traj)
    d_to_data = np.sqrt(((traj[-1][:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
    return traj, d_to_data


def add_trajectory_density(ax, traj, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    segments = [traj[:, i, :] for i in range(traj.shape[1])]
    colors = cmap(np.linspace(0.35, 0.95, traj.shape[1]))
    colors[:, 3] = 0.05
    lines = LineCollection(segments, colors=colors, linewidths=0.8)
    ax.add_collection(lines)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / "assets" / "quad-trajectories.png"
    out.parent.mkdir(exist_ok=True)

    X = make_swiss_roll(1500, 0.05)
    seeds = {"DDPM": 934, "EDM": 720, "FM": 672, "EqM": 997}
    models = {name: train_one(name, X, steps=2500, seed=seeds[name]) for name in PARAMETRIZATIONS}

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=100, facecolor="white")
    for ax, name in zip(axes.ravel(), PARAMETRIZATIONS):
        traj, d_to_data = sample_trajectories(name, models[name], X, n=200, K=100, dt=0.04, seed=42)
        pct = (d_to_data < 0.15).mean() * 100
        ax.set_facecolor("white")
        ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, c="#0a84ff", edgecolors="none")
        add_trajectory_density(ax, traj, CMAPS[name])
        ax.set_title(f"{name} - K=100 steps - {pct:.0f}% within ε", fontsize=14)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.22, linewidth=0.8)

    fig.suptitle("Sample trajectories on swiss roll. The shape is the basin geometry.", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
