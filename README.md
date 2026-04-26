# The Geometry of Noise — explorable companion notebook

An interactive marimo notebook companion to **Sahraee-Ardakan, Delbracio, Milanfar (2026)**, *"The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning"* — [arXiv:2602.18428](https://arxiv.org/abs/2602.18428).

Trains all 4 parametrizations from the paper's Table 1 (DDPM, EDM, Flow Matching, Equilibrium Matching) from scratch in pure numpy on a 2D swiss roll, and reproduces the paper's structural instability claim:

- At K=100, dt=0.04: **FM converges 100%, DDPM 0%**.
- Falsification panel maps the stability basin of all 4 parametrizations across (K, dt).
- Basin Explorer: drag sliders, watch the basin shape live.

Built for the **alphaXiv × marimo notebook competition**, April 2026.

## Run locally

```bash
uv venv && source .venv/bin/activate
uv pip install marimo numpy scipy matplotlib
marimo run notebook.py
```

## Run in your browser

[Open in molab](MOLAB_URL_PLACEHOLDER) — runs entirely in WASM, no install required.

## License

MIT
