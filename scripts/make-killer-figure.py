import pathlib

import matplotlib.pyplot as plt
import numpy as np


PARAMETRIZATIONS = ["DDPM", "EDM", "FM", "EqM"]
TARGET_COEFFS = {
    "DDPM": lambda t: (np.zeros_like(t), np.ones_like(t)),
    "EDM": lambda t: (np.ones_like(t), np.zeros_like(t)),
    "FM": lambda t: (-np.ones_like(t), np.ones_like(t)),
    "EqM": lambda t: (-t, t),
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
    p = init_mlp(seed=seed)
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


def sample_run(name, model, X, n=300, K=100, dt=0.04, seed=42):
    g = np.random.default_rng(seed)
    u = g.standard_normal((n, 2)) * 1.0
    traj_idx = g.choice(n, size=30, replace=False)
    traj = [u[traj_idx].copy()]
    for _ in range(K):
        pred, _ = forward(model, u)
        u = u + dt * step_dir(name, pred, u)
        u = np.clip(u, -10, 10)
        traj.append(u[traj_idx].copy())
    d_to_data = np.sqrt(((u[:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
    return u, np.array(traj), (d_to_data < 0.15).mean() * 100


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    (root / "assets").mkdir(exist_ok=True)
    X = make_swiss_roll(1500, 0.05)
    models = {}
    seeds = {"DDPM": 934, "EDM": 720, "FM": 672, "EqM": 997}

    for name in PARAMETRIZATIONS:
        models[name] = train_one(name, X, steps=2500, seed=seeds[name])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=100)
    for ax, name in zip(axes.ravel(), PARAMETRIZATIONS):
        u, traj, pct = sample_run(name, models[name], X)
        print(f"{name}: {pct:.0f}%")
        ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.25, c="#0a84ff")
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.45, alpha=0.35, c="grey")
        ax.scatter(u[:, 0], u[:, 1], s=9, c="#ff3b30", alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f"{name} — {pct:.0f}% conv")
        ax.grid(alpha=0.15)

    fig.suptitle("Same K, same dt, only the parametrization differs")
    fig.tight_layout()
    fig.savefig(root / "assets" / "killer-figure.png", bbox_inches="tight")
    print(root / "assets" / "killer-figure.png")


if __name__ == "__main__":
    main()
