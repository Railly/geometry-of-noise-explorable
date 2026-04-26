1. I built a marimo notebook for the alphaXiv x marimo competition: an explorable companion to "The Geometry of Noise" (arXiv:2602.18428).

Live notebook: {MOLAB_URL}

The point: show why removing noise conditioning works for some parametrizations and breaks badly for others.

2. The notebook trains DDPM, EDM, Flow Matching, and Equilibrium Matching from scratch in pure numpy on a 2D swiss roll.

Same data. Same MLP. Same integrator.

Only the parametrization changes.

3. The kill-shot:

At K=100, dt=0.04, seed=42 on the swiss roll:

- FM converges 100%
- DDPM converges 0%

That gap is the paper's stability claim made visible.

4. DDPM is not "worse" because the network is bigger/smaller or because the dataset changed.

It is unstable here because noise prediction becomes a high-gain amplifier when you remove explicit noise conditioning.

Velocity/restoration parametrizations have a much wider stable basin.

5. The notebook includes a Basin Explorer.

Drag K, dt, and seed. Toggle DDPM/EDM/FM/EqM.

You can watch DDPM collapse at the same settings where FM lands cleanly on the manifold.

6. I also added a falsification panel: 4 x 8 heatmaps over (K, dt).

The goal is not just to show one cherry-picked sample path.

It maps the basin of stability so the claim can be poked, broken, and compared interactively.

7. This is why I like marimo for research artifacts.

The notebook is reactive, git-friendly, Pythonic, and usable as a small app.

Papers explain claims. Interactive notebooks let people test the claims.

8. Thanks @marimo_io @alphaXiv for the competition. Notebook source: github.com/Railly/geometry-of-noise-explorable (push pending).
