# The Geometry of Noise: Explorable Companion

Interactive marimo notebook for stress-testing diffusion parametrizations on a 2D swiss roll.

![](assets/killer-figure-v2.png)

This project recreates the core instability experiment from Sahraee-Ardakan, Delbracio, and Milanfar (2026) in a browser-ready marimo notebook, training DDPM, EDM, Flow Matching, and Equilibrium Matching from scratch in pure NumPy and exposing the convergence basin so the paper's claim can be inspected, falsified, and replayed.

[![marimo](https://img.shields.io/badge/marimo-notebook-FF6B6B)](https://marimo.io/)
[![NumPy](https://img.shields.io/badge/NumPy-2D%20experiments-013243)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![arXiv](https://img.shields.io/badge/arXiv-2602.18428-b31b1b.svg)](https://arxiv.org/abs/2602.18428)

## Table of Contents

- [Quick Try](#quick-try)
- [The Kill-Shot](#the-kill-shot)
- [What We Built](#what-we-built)
- [Run Locally](#run-locally)
- [Cite](#cite)
- [License](#license)

## Quick Try

- [Open the notebook in molab](https://molab.marimo.io/notebooks/nb_B3Ns3kjFJaT5ayKniGdJgx)
- [Watch the kill-shot GIF](assets/kill-shot.gif)

## The Kill-Shot

At `K=100` and `dt=0.04`, the parametrizations split hard:

| Parametrization | Convergence |
| --- | ---: |
| Flow Matching | 100% |
| DDPM | 0% |
| EDM | 75% |
| Equilibrium Matching | 61% |

## What We Built

- Four parametrizations from scratch: DDPM, EDM, Flow Matching, and Equilibrium Matching
- Basin explorer for sweeping `K` and `dt`
- Falsification panel for comparing stability claims
- Gallery of convergence and failure modes
- Kill-shot GIF for the headline result

## Run Locally

```bash
uv venv
source .venv/bin/activate
uv pip install marimo numpy scipy matplotlib
marimo run notebook.py
```

## Cite

```bibtex
@misc{sahraeeardakan2026geometrynoise,
  title = {The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning},
  author = {Sahraee-Ardakan, Mojtaba and Delbracio, Mauricio and Milanfar, Peyman},
  year = {2026},
  eprint = {2602.18428},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2602.18428}
}
```

## License

MIT
