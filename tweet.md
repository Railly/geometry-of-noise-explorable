# Launch Tweet Thread — alphaXiv x marimo competition

## Tweet 1/8 (the hook — most important)

```
Diffusion is sold as robust sampling. On a 2D swiss roll, that depends on coordinates.

In my marimo notebook: at K=100, dt=0.04, FM returns 100% within ε=0.15 of the manifold. DDPM returns 0%.

Try the WASM molab: {MOLAB_URL}
```
(225 chars)

## Tweet 2/8

```
The paradox in The Geometry of Noise: the same probability path can become stable or unstable after a harmless-looking reparametrization.

So the question is not only "what distribution?" It is "what geometry did the sampler inherit?"
```
(238 chars)

## Tweet 3/8

```
We built a notebook companion that trains 4 parametrizations from scratch on a 2D swiss roll: DDPM, EDM, FM, EqM.

Same data manifold, same toy setup, four learned vector fields pulling samples back differently.
```
(212 chars)

## Tweet 4/8 (with image: assets/killer-figure-v2.png)

```
The kill-shot is the K=100, dt=0.04 panel.

FM: 100% convergence within ε=0.15 of the swiss-roll manifold.
DDPM: 0%.

Same dataset. Same compute scale. Different parametrization.
```
(187 chars — leaves room for image)

## Tweet 5/8

```
The notebook adds a falsification panel: 32 runs across (K, dt), not one cherry-picked trajectory.

It maps where each parametrization stays stable, where it breaks, and where the paper's structural-instability claim survives contact with sweeps.
```
(250 chars)

## Tweet 6/8

```
Technical constraint: no PyTorch, no JAX, no install.

The models train in pure numpy, including the MLPs, losses, and from-scratch backprop. Then the whole thing runs in marimo's WASM molab so reviewers can rerun it in-browser.
```
(232 chars)

## Tweet 7/8

```
Credit to Sahraee-Ardakan, Delbracio, and Milanfar for The Geometry of Noise (arXiv:2602.18428).

This is my submission to the @alphaXiv x @marimo_io notebook competition: a reproducible companion, not just a summary.
```
(221 chars)

## Tweet 8/8

```
If you want to test the claim, run the notebook. Change K, dt, ε, or the manifold tolerance and watch the stability map move.

Repo: github.com/Railly/geometry-of-noise-explorable
Molab: {MOLAB_URL}

Built by Hunter (@raillyhugo), AI engineer at Clerk.
```
(247 chars)

---

## Posting checklist

- [ ] Replace `{MOLAB_URL}` (2 places: tweet 1 + tweet 8)
- [ ] Tweet 4 attach `assets/killer-figure-v2.png` OR `assets/kill-shot.gif` if generated
- [ ] Post all 8 sequentially as a thread (reply to your own previous tweet)
- [ ] Tag @marimo_io and @alphaXiv only in tweet 7 (not earlier — looks spammy)
- [ ] After posting: drop link in marimo Discord #molab-competition channel

## Quote-tweet candidates (extra distribution)

- @milanfar (Peyman Milanfar — paper coauthor, Google, very active on X)
- @marimo_io
- @alphaXiv
