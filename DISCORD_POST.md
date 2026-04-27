https://molab.marimo.io/notebooks/nb_RBaLdF27xB6V5D84XkWYz4

Hi #molab-competition, my submission: a companion to *The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning* (Sahraee-Ardakan, Delbracio, Milanfar 2026, arXiv:2602.18428).

It trains 4 parametrizations from scratch on a 2D swiss roll (DDPM, EDM, FM, EqM) and reproduces the kill-shot: at K=100, dt=0.04, FM converges 100% within ε of the manifold while DDPM converges 0%. Same data, same network, same sampler.

The extension is a falsification panel: 32 runs across (K, dt) that map the stability basin of each parametrization. Pure numpy in WASM, no install required.

Repo: https://github.com/Railly/geometry-of-noise-explorable

Feedback welcome on framing and whether the falsification panel makes the gap easier to trust.

— Hunter / @raillyhugo
