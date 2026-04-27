"""Reproduce the autonomous diffusion claim from arXiv 2602.18428.

This module contains the small NumPy MLP, training targets, autonomous step
rules, and sampler used to compare DDPM, EDM, FM, and EqM on a 2D swiss roll.
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np


PARAMETRIZATIONS = ["DDPM", "EDM", "FM", "EqM"]

TARGET_COEFFS = {
    "DDPM": lambda t: (np.zeros_like(t), np.ones_like(t)),
    "EDM": lambda t: (np.ones_like(t), np.zeros_like(t)),
    "FM": lambda t: (-np.ones_like(t), np.ones_like(t)),
    "EqM": lambda t: (-t, t),
}


def init_mlp(in_dim, hidden, out_dim, seed):
    g = np.random.default_rng(seed)
    return {
        "W1": g.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
        "b1": np.zeros(hidden),
        "W2": g.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden),
        "b2": np.zeros(hidden),
        "W3": g.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden),
        "b3": np.zeros(out_dim),
    }


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_deriv(x):
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x**3)
    t = np.tanh(u)
    du_dx = c * (1.0 + 3.0 * 0.044715 * x**2)
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
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(p, grads, lr):
    return {k: p[k] - lr * grads[k] for k in p}


def schedule(t):
    return 1.0 - t, t


def step_dir(name, pred, u):
    if name == "DDPM":
        return -pred
    if name == "EDM":
        return pred - u
    if name in ("FM", "EqM"):
        return -pred
    raise ValueError(f"unknown parametrization: {name}")


def train_one(name, X, steps, batch, lr, seed):
    if name not in TARGET_COEFFS:
        raise ValueError(f"unknown parametrization: {name}")

    rng = np.random.default_rng(seed)
    p = init_mlp(2, 64, 2, seed)
    losses = []
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
        losses.append((err**2).mean())
        dout = 2.0 * err / pred.shape[1]
        grads = backward(p, cache, dout)
        p = sgd_step(p, grads, lr)

    return p, np.array(losses)


def sample(model, name, n, K, dt, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, 2))
    traj = [u.copy()]

    for _ in range(K):
        pred, _ = forward(model, u)
        u = u + dt * step_dir(name, pred, u)
        u = np.clip(u, -10.0, 10.0)
        traj.append(u.copy())

    return np.array(traj)


def near_manifold(samples, X, eps):
    pts = samples[-1] if samples.ndim == 3 else samples
    d = _manifold_dist(pts, X)
    converged_pct = (d < eps).mean() * 100.0
    diverged_pct = (np.linalg.norm(pts, axis=1) > 5.0).mean() * 100.0
    return converged_pct, diverged_pct


def _make_swiss_roll(n, jitter, seed):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    pts = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 12.0
    pts += jitter * rng.standard_normal(pts.shape)
    return pts


def _manifold_dist(pts, X):
    return np.sqrt(((pts[:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)


if __name__ == "__main__":
    X = _make_swiss_roll(1500, 0.05, seed=0)
    seeds = {"DDPM": 101, "EDM": 202, "FM": 303, "EqM": 404}

    for name in PARAMETRIZATIONS:
        model, _ = train_one(name, X, steps=2500, batch=256, lr=2e-3, seed=seeds[name])
        traj = sample(model, name, n=400, K=100, dt=0.04, seed=42)
        converged_pct, _ = near_manifold(traj, X, eps=0.15)
        mean_dist = _manifold_dist(traj[-1], X).mean()
        print(f"| {name} | {converged_pct:.2f} | {mean_dist:.4f} |")
