# Launch tweet thread

## Tweet 1/8 (the hook)

```
Diffusion is sold as robust sampling. On a 2D swiss roll, that depends on coordinates.

In my marimo notebook: at K=100, dt=0.04, FM lands 100% within ε=0.15 of the manifold. DDPM lands 0%.

Try the WASM molab: https://molab.marimo.io/notebooks/nb_B3Ns3kjFJaT5ayKniGdJgx
```

## Tweet 2/8

```
The paradox in The Geometry of Noise: the same probability path can become stable or unstable after a harmless reparametrization.

So the question is not just "what distribution?" It's "what geometry did the sampler inherit?"
```

## Tweet 3/8

```
The notebook trains 4 parametrizations from scratch on the same 2D swiss roll: DDPM, EDM, FM, EqM.

Same data manifold, same toy setup. Four learned vector fields pulling samples back differently.
```

## Tweet 4/8 (attach: assets/kill-shot.gif or assets/killer-figure-v2.png)

```
The kill-shot is the K=100, dt=0.04 panel.

FM: 100% convergence within ε=0.15 of the swiss-roll manifold.
DDPM: 0%.

Same dataset. Same compute scale. Different parametrization.
```

## Tweet 5/8

```
The notebook adds a falsification panel: 32 runs across (K, dt), not one cherry-picked trajectory.

It maps where each parametrization stays stable, where it breaks, and how sharp the boundary is.
```

## Tweet 6/8

```
Technical constraint: no PyTorch, no JAX, no install.

The models train in pure numpy, including the MLPs, losses, and from-scratch backprop. The whole thing runs in marimo's WASM molab so reviewers can rerun it in-browser.
```

## Tweet 7/8

```
Credit to Sahraee-Ardakan, Delbracio, and Milanfar for The Geometry of Noise (arXiv:2602.18428).

This is my submission to the @alphaXiv x @marimo_io notebook competition: a reproducible companion, not a summary.
```

## Tweet 8/8

```
If you want to test the claim, run the notebook. Change K, dt, ε, or the manifold tolerance and watch the stability map move.

Repo: github.com/Railly/geometry-of-noise-explorable
Molab: https://molab.marimo.io/notebooks/nb_B3Ns3kjFJaT5ayKniGdJgx

Built by Hunter (@raillyhugo), AI engineer at Clerk.
```

---

## Posting checklist

- [ ] Post 1: drop the molab link, no thread reply yet
- [ ] Post 2-8: each as reply to the previous
- [ ] Tweet 4: attach `assets/kill-shot.gif` (preferred) or `assets/killer-figure-v2.png`
- [ ] Tag @marimo_io and @alphaXiv only in tweet 7 (avoids spammy first impression)
- [ ] After posting: drop the link in marimo Discord #molab-competition channel

## Quote-tweet candidates (extra distribution)

- @milanfar (Peyman Milanfar, paper coauthor at Google, active on X)
- @marimo_io
- @alphaXiv
