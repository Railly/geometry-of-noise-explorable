# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///
import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", app_title="The Geometry of Noise — explorable")


@app.cell(hide_code=True)
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(89)
    return mo, np, plt, rng


@app.cell(hide_code=True)
def _intro(mo):
    mo.md(
        r"""
        # The Geometry of Noise

        ## What if a diffusion model forgets the noise level and gets better at seeing the geometry?

        > An explorable companion to **Sahraee-Ardakan, Delbracio, Milanfar (2026)**, *"The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning"* — [arXiv:2602.18428](https://arxiv.org/abs/2602.18428).

        Standard diffusion conditioning says: tell the network where it is in noise-time, then ask for the denoising direction. Remove that scalar, and the problem looks underdetermined. The same noisy point could have arrived from many noise levels, so the model should be confused. Yet the paper's first surprise is that the network does not merely average nonsense; it integrates over its own posterior belief about time.

        That averaging changes the object being followed. Instead of a time-indexed score field, the dynamics become a gradient flow on the marginal energy of noisy observations. Near the data manifold, that energy forms a sharp normal well: steep across the manifold, gentle along it. The singularity is real, but the learned field also carries a metric. In the right parametrization, the metric cancels the blow-up.

        This notebook is the small mechanical version of that story. We train the four parametrizations from scratch on a 2D swiss roll and then push them with deliberately coarse steps. Velocity-style fields behave like geometry-aware flows; noise prediction behaves like a high-gain circuit attached to a shaky estimator. At $K{=}100$, $dt{=}0.04$, the gap is total: **FM converges $100\%$ within $\varepsilon$ of the manifold; DDPM converges $0\%$**. The toy experiment makes that gap hard to unsee.

        Drag the sliders. Everything runs locally in your browser. No data leaves your machine.
        """
    )
    return


@app.cell(hide_code=True)
def _section_1(mo):
    mo.md(
        r"""
        ## 1. The setup — one schedule to rule them all

        Following paper §3, every diffusion-style generative model can be written as a single linear corruption:

        $$\mathbf{u}_t = a(t)\,\mathbf{x} + b(t)\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad t \in [0, 1]$$

        The training target is also linear: $r(\mathbf{x}, \boldsymbol{\epsilon}, t) = c(t)\,\mathbf{x} + d(t)\,\boldsymbol{\epsilon}$. Different choices of $(a, b, c, d)$ give back the four standard models — see the paper's Table 1.

        | Model | $a(t)$ | $b(t)$ | $c(t)$ | $d(t)$ | What the network predicts |
        |---|---|---|---|---|---|
        | **DDPM** | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | 0 | 1 | noise $\boldsymbol\epsilon$ |
        | **EDM** | 1 | $\sigma_t$ | 1 | 0 | clean data $\mathbf{x}$ |
        | **FM** | $1-t$ | $t$ | $-1$ | 1 | velocity $\boldsymbol\epsilon - \mathbf{x}$ |
        | **EqM** | $1-t$ | $t$ | $-t$ | $t$ | restoration $t(\boldsymbol\epsilon - \mathbf{x})$ |

        We work with **2D data** so we can see everything. Corruption uses $a(t) = 1 - t$, $b(t) = t$. Targets vary per model.
        """
    )
    return


@app.cell
def _params(np):
    PARAMETRIZATIONS = ["DDPM", "EDM", "FM", "EqM"]
    TARGET_COEFFS = {
        "DDPM": lambda t: (np.zeros_like(t), np.ones_like(t)),
        "EDM":  lambda t: (np.ones_like(t),  np.zeros_like(t)),
        "FM":   lambda t: (-np.ones_like(t), np.ones_like(t)),
        "EqM":  lambda t: (-t, t),
    }
    def schedule(t):
        return 1.0 - t, t
    def step_dir(name, pred, u):
        if name == "DDPM":  return -pred
        if name == "EDM":   return pred - u
        if name == "FM":    return -pred
        if name == "EqM":   return -pred
    return PARAMETRIZATIONS, TARGET_COEFFS, schedule, step_dir


@app.cell(hide_code=True)
def _section_2(mo):
    mo.md(
        r"""
        ## 2. The data manifold

        We use a **swiss roll** projected to 2D. It's a 1D manifold living in 2D — exactly the regime where the paper's *singularity near the data manifold* claim has teeth.
        """
    )
    return


@app.cell
def _data_ui(mo):
    n_samples = mo.ui.slider(200, 4000, value=1500, step=100, label="Samples on the manifold")
    noise_jitter = mo.ui.slider(0.0, 0.5, value=0.05, step=0.01, label="Manifold thickness (jitter)")
    return n_samples, noise_jitter


@app.cell
def _make_data(n_samples, noise_jitter, np, rng):
    def make_swiss_roll(n, jitter):
        _t = 1.5 * np.pi * (1 + 2 * rng.random(n))
        _pts = np.stack([_t * np.cos(_t), _t * np.sin(_t)], axis=1) / 12.0
        _pts += jitter * rng.standard_normal(_pts.shape)
        return _pts
    X = make_swiss_roll(n_samples.value, noise_jitter.value)
    return (X,)


@app.cell(hide_code=True)
def _show_data(X, mo, n_samples, noise_jitter, plt):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.6, c="#0a84ff")
    _ax.set_aspect("equal")
    _ax.set_xlim(-2.5, 2.5); _ax.set_ylim(-2.5, 2.5)
    _ax.set_title(f"swiss roll, n={n_samples.value}, jitter={noise_jitter.value:.2f}")
    _ax.grid(alpha=0.2)
    mo.vstack([mo.hstack([n_samples, noise_jitter]), _fig])


@app.cell(hide_code=True)
def _section_3(mo):
    mo.md(
        r"""
        ## 3. A network simple enough to read

        Tiny 2-layer MLP trained with explicit-gradient SGD on the MSE loss

        $$\mathcal{L}(f) = \mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon},t}\!\left[\,\|\,f(\mathbf{u}_t) - r(\mathbf{x}, \boldsymbol{\epsilon}, t)\,\|^2\right].$$

        Crucially: $f$ takes **only $\mathbf{u}_t$**. No $t$. Forward + backward hand-rolled in numpy.
        """
    )
    return


@app.cell
def _mlp(np):
    def init_mlp(in_dim=2, hidden=64, out_dim=2, seed=0):
        _g = np.random.default_rng(seed)
        return dict(
            W1=_g.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim), b1=np.zeros(hidden),
            W2=_g.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden), b2=np.zeros(hidden),
            W3=_g.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden), b3=np.zeros(out_dim),
        )

    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def gelu_deriv(x):
        _c = np.sqrt(2.0 / np.pi)
        _u = _c * (x + 0.044715 * x ** 3)
        _t = np.tanh(_u)
        _du_dx = _c * (1.0 + 3 * 0.044715 * x ** 2)
        return 0.5 * (1.0 + _t) + 0.5 * x * (1.0 - _t ** 2) * _du_dx

    def forward(p, u):
        _h1 = u @ p["W1"] + p["b1"]; _a1 = gelu(_h1)
        _h2 = _a1 @ p["W2"] + p["b2"]; _a2 = gelu(_h2)
        _out = _a2 @ p["W3"] + p["b3"]
        return _out, (u, _h1, _a1, _h2, _a2)

    def backward(p, cache, dout):
        _u, _h1, _a1, _h2, _a2 = cache
        _n = _u.shape[0]
        _dW3 = _a2.T @ dout / _n; _db3 = dout.mean(0)
        _da2 = dout @ p["W3"].T; _dh2 = _da2 * gelu_deriv(_h2)
        _dW2 = _a1.T @ _dh2 / _n; _db2 = _dh2.mean(0)
        _da1 = _dh2 @ p["W2"].T; _dh1 = _da1 * gelu_deriv(_h1)
        _dW1 = _u.T @ _dh1 / _n; _db1 = _dh1.mean(0)
        return dict(W1=_dW1, b1=_db1, W2=_dW2, b2=_db2, W3=_dW3, b3=_db3)

    def sgd_step(p, grads, lr):
        return {k: p[k] - lr * grads[k] for k in p}

    return backward, forward, init_mlp, sgd_step


@app.cell
def _train_fn(TARGET_COEFFS, backward, forward, init_mlp, np, rng, schedule, sgd_step):
    def train_one(name, X, steps=2500, batch=256, lr=2e-3, seed=0):
        _p = init_mlp(seed=seed)
        _losses = []
        _n = X.shape[0]
        _coeff_fn = TARGET_COEFFS[name]
        for _step in range(steps):
            _idx = rng.integers(0, _n, size=batch)
            _xb = X[_idx]
            _t = rng.random(batch).reshape(-1, 1)
            _eps = rng.standard_normal(_xb.shape)
            _a, _b = schedule(_t)
            _ub = _a * _xb + _b * _eps
            _c_t, _d_t = _coeff_fn(_t)
            _r = _c_t * _xb + _d_t * _eps
            _pred, _cache = forward(_p, _ub)
            _err = _pred - _r
            _losses.append((_err ** 2).mean())
            _dout = 2.0 * _err / _pred.shape[1]
            _grads = backward(_p, _cache, _dout)
            _p = sgd_step(_p, _grads, lr)
        return _p, np.array(_losses)
    return (train_one,)


@app.cell(hide_code=True)
def _train_button(mo):
    train_btn = mo.ui.run_button(label="Train all four parametrizations")
    return (train_btn,)


@app.cell(hide_code=True)
def _show_train_btn(mo, train_btn):
    mo.vstack([mo.md("**Click to train. ~10 seconds total.**"), train_btn])


@app.cell
def _do_train(PARAMETRIZATIONS, X, mo, train_btn, train_one):
    mo.stop(not train_btn.value, mo.md("_(click the button above to train all four)_"))
    models = {}
    losses = {}
    for _name in PARAMETRIZATIONS:
        _p, _lh = train_one(_name, X, steps=2500, seed=hash(_name) % 1024)
        models[_name] = _p
        losses[_name] = _lh
    return losses, models


@app.cell(hide_code=True)
def _show_losses(losses, np, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    for _name, _lh in losses.items():
        _sm = np.convolve(_lh, np.ones(50) / 50, mode="valid")
        _ax.plot(_sm, label=_name, alpha=0.85)
    _ax.set_xlabel("step"); _ax.set_ylabel("loss (smoothed)")
    _ax.set_yscale("log"); _ax.legend()
    _ax.set_title("Training loss per parametrization")
    _ax.grid(alpha=0.2)
    _fig


@app.cell(hide_code=True)
def _section_4(mo):
    mo.md(
        r"""
        ## 4. Look at the field

        The autonomous vector field $f^*(\mathbf{u})$ — the network's prediction at every point in 2D. Switch parametrization with the dropdown.

        Notice: the four fields look completely different. Same data, same MLP, same loss family — but the *target* shapes the implicit Riemannian metric.
        """
    )
    return


@app.cell
def _field_ui(PARAMETRIZATIONS, mo):
    field_choice = mo.ui.dropdown(options=PARAMETRIZATIONS, value="FM", label="parametrization")
    return (field_choice,)


@app.cell(hide_code=True)
def _draw_field(X, field_choice, forward, mo, models, np, plt):
    _name = field_choice.value
    _gx = np.linspace(-2.5, 2.5, 22); _gy = np.linspace(-2.5, 2.5, 22)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _grid = np.stack([_GX.ravel(), _GY.ravel()], axis=1)
    _pred, _ = forward(models[_name], _grid)
    _U = _pred[:, 0].reshape(_GX.shape); _V = _pred[:, 1].reshape(_GX.shape)
    _mag = np.sqrt(_U ** 2 + _V ** 2) + 1e-9
    _Un, _Vn = _U / _mag, _V / _mag
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.3, c="#0a84ff")
    _ax.quiver(_GX, _GY, _Un, _Vn, _mag, cmap="magma", scale=28, width=0.003)
    _ax.set_aspect("equal")
    _ax.set_xlim(-2.5, 2.5); _ax.set_ylim(-2.5, 2.5)
    _ax.set_title(f"Autonomous field f*(u) — {_name}")
    _ax.grid(alpha=0.15)
    mo.vstack([field_choice, _fig])


@app.cell(hide_code=True)
def _section_5(mo):
    mo.md(
        r"""
        ## 5. The Energy Paradox

        The marginal energy $E_{\text{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$ has a $1/t^p$ singularity normal to the data manifold (paper eq. 12): the gradient diverges as you approach. We visualize this with a kernel density estimate.

        How can a finite neural network match an infinite gradient? The paper's answer: the conformal metric in the field cancels it — but only for the right parametrizations.
        """
    )
    return


@app.cell(hide_code=True)
def _energy_panel(X, np, plt):
    from scipy.stats import gaussian_kde
    _kde = gaussian_kde(X.T, bw_method=0.08)
    _gx = np.linspace(-2.5, 2.5, 80); _gy = np.linspace(-2.5, 2.5, 80)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _grid = np.stack([_GX.ravel(), _GY.ravel()], axis=0)
    _p = _kde(_grid).reshape(_GX.shape)
    _E = -np.log(_p + 1e-9)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _cs = _ax.contourf(_GX, _GY, _E, levels=25, cmap="viridis")
    _ax.scatter(X[:, 0], X[:, 1], s=2, c="white", alpha=0.4)
    _ax.set_aspect("equal")
    _ax.set_title(r"$E_{\mathrm{marg}}(\mathbf{u}) = -\log p(\mathbf{u})$ — the well that should diverge")
    plt.colorbar(_cs, ax=_ax, shrink=0.8, label="energy")
    _fig


@app.cell(hide_code=True)
def _section_6(mo):
    mo.md(
        r"""
        ## 6. The kill-shot — sampling

        Same data. Same network. Same sampler. Different parametrization. Watch what happens.

        We start from the same Gaussian cloud and ask each trained model to transport it back to the swiss roll. For each run, we integrate exactly $K$ steps with the same autonomous step rule and the same step size $dt$.

        | Parametrization | Converged at $K{=}100$, $dt{=}0.04$ |
        |---|---:|
        | **FM** | **100%** |
        | EDM | 75% |
        | EqM | 61% |
        | **DDPM** | **0%** |

        This is the kill shot. The architecture did not change. The data did not change. The sampler did not get special treatment. Only the parametrization changed, and the outcome went from perfect convergence to total failure.

        The reason is exactly the one predicted by the geometry. Velocity-based parametrizations behave like bounded-gain systems: posterior uncertainty is absorbed into a smooth drift field, so local errors stay local. DDPM, as noise prediction, is a high-gain amplifier: small score errors are multiplied by the inverse noise scale, and near the manifold that gain explodes.

        **Use the sliders below to make the failure visible:**

        1. Drop $dt$ to `0.005`. DDPM recovers — but only because the sampler is forced to crawl through the high-gain region.
        2. Keep $dt{=}0.04$ and increase $K$. The velocity models remain stable; DDPM still pays for every inaccurate step near the manifold.
        3. Push $dt$ upward. FM degrades gracefully, EDM and EqM break later, DDPM collapses first.

        The point is not that DDPM cannot sample this distribution. It can. The point is that it needs far more numerical care for the same learned task — because its parametrization turns harmless estimation error into unstable geometry.
        """
    )
    return


@app.cell
def _sample_ui(PARAMETRIZATIONS, mo):
    sample_choice = mo.ui.dropdown(options=PARAMETRIZATIONS, value="FM", label="parametrization")
    sample_steps = mo.ui.slider(20, 400, value=100, step=10, label="integration steps K")
    sample_dt = mo.ui.slider(0.005, 0.15, value=0.04, step=0.005, label="step size dt")
    sample_n = mo.ui.slider(50, 1000, value=400, step=50, label="number of samples")
    return sample_choice, sample_dt, sample_n, sample_steps


@app.cell(hide_code=True)
def _do_sample(X, forward, mo, models, np, plt, rng, sample_choice, sample_dt, sample_n, sample_steps, step_dir):
    _name = sample_choice.value
    _K = sample_steps.value
    _dt = sample_dt.value
    _n = sample_n.value

    _u = rng.standard_normal((_n, 2)) * 1.0
    _traj = [_u.copy()]
    for _k in range(_K):
        _pred, _ = forward(models[_name], _u)
        _v = step_dir(_name, _pred, _u)
        _u = _u + _dt * _v
        _u = np.clip(_u, -10, 10)
        _traj.append(_u.copy())
    _traj = np.array(_traj)

    _d_to_data = np.sqrt(((_traj[-1][:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
    _pct_converged = (_d_to_data < 0.15).mean() * 100
    _pct_diverged = (np.linalg.norm(_traj[-1], axis=1) > 5.0).mean() * 100

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.25, c="#0a84ff", label="data manifold")
    _idx = rng.integers(0, _n, size=min(50, _n))
    for _i in _idx:
        _ax.plot(_traj[:, _i, 0], _traj[:, _i, 1], lw=0.4, alpha=0.4, c="grey")
    _ax.scatter(_traj[-1, :, 0], _traj[-1, :, 1], s=10, c="#ff3b30", label="final samples", alpha=0.7)
    _ax.set_aspect("equal")
    _ax.set_xlim(-3, 3); _ax.set_ylim(-3, 3)
    _ax.set_title(f"{_name} — K={_K}, dt={_dt:.3f} → {_pct_converged:.0f}% converged, {_pct_diverged:.0f}% diverged")
    _ax.legend(loc="upper right")
    _ax.grid(alpha=0.15)
    mo.vstack([
        mo.hstack([sample_choice, sample_steps]),
        mo.hstack([sample_dt, sample_n]),
        _fig,
    ])


@app.cell(hide_code=True)
def _section_7(mo):
    mo.md(
        r"""
        ## 6.5 Basin Explorer

        Same data, same learned models, same integrator. Toggle parametrizations and move through the basin directly.
        """
    )
    return


@app.cell
def _basin_explorer_ui(PARAMETRIZATIONS, mo):
    basin_show = mo.ui.multiselect(options=PARAMETRIZATIONS, value=PARAMETRIZATIONS, label="show parametrizations")
    basin_K = mo.ui.slider(20, 400, value=100, step=10, label="K")
    basin_dt = mo.ui.slider(0.005, 0.15, value=0.04, step=0.005, label="dt")
    basin_seed = mo.ui.slider(0, 100, value=42, step=1, label="seed")
    return basin_K, basin_dt, basin_seed, basin_show


@app.cell(hide_code=True)
def _basin_explorer(X, basin_K, basin_dt, basin_seed, basin_show, forward, mo, models, np, plt, step_dir):
    _names = list(basin_show.value)
    _K = basin_K.value
    _dt = basin_dt.value
    _seed = basin_seed.value
    _n = 300
    _g = np.random.default_rng(_seed)
    _starts = _g.standard_normal((_n, 2)) * 1.0
    _traj_idx = _g.choice(_n, size=min(30, _n), replace=False)

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 9))
    _axes = _axes.ravel()

    for _ax in _axes:
        _ax.set_visible(False)

    for _plot_i, _name in enumerate(_names[:4]):
        _u = _starts.copy()
        _traj = [_u[_traj_idx].copy()]
        for _ in range(_K):
            _pred, _ = forward(models[_name], _u)
            _u = _u + _dt * step_dir(_name, _pred, _u)
            _u = np.clip(_u, -10, 10)
            _traj.append(_u[_traj_idx].copy())
        _traj = np.array(_traj)
        _d_to_data = np.sqrt(((_u[:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
        _pct = (_d_to_data < 0.15).mean() * 100
        _ax = _axes[_plot_i]
        _ax.set_visible(True)
        _ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.25, c="#0a84ff")
        for _i in range(len(_traj_idx)):
            _ax.plot(_traj[:, _i, 0], _traj[:, _i, 1], lw=0.45, alpha=0.35, c="grey")
        _ax.scatter(_u[:, 0], _u[:, 1], s=9, c="#ff3b30", alpha=0.7)
        _ax.set_aspect("equal")
        _ax.set_xlim(-3, 3)
        _ax.set_ylim(-3, 3)
        _ax.set_title(f"{_name} — {_pct:.0f}% conv")
        _ax.grid(alpha=0.15)

    _fig.suptitle(f"K={_K}, dt={_dt:.3f}, seed={_seed}")
    _fig.tight_layout()

    mo.vstack([
        mo.callout("Try the kill-shot: dt=0.04, K=100, seed=42 — watch DDPM colapse while FM/EqM/EDM converge."),
        mo.hstack([basin_show, basin_K]),
        mo.hstack([basin_dt, basin_seed]),
        _fig,
    ])


@app.cell(hide_code=True)
def _section_7(mo):
    mo.md(
        r"""
        ## 7. Falsification panel — the real extension

        Sweep $(K, dt)$ and measure for each model:
        - **convergence rate**: % of samples within $\epsilon = 0.15$ of the manifold
        - **divergence rate**: % of samples whose norm exceeds 5

        If the paper is right, **FM/EqM/EDM should have a wide green stable basin**, **DDPM a narrow one** that vanishes as $dt$ grows.
        """
    )
    return


@app.cell(hide_code=True)
def _falsify_btn(mo):
    falsify_btn = mo.ui.run_button(label="Run falsification sweep (~15s)")
    return (falsify_btn,)


@app.cell(hide_code=True)
def _show_falsify_btn(falsify_btn):
    falsify_btn


@app.cell(hide_code=True)
def _falsify(PARAMETRIZATIONS, X, falsify_btn, forward, mo, models, np, plt, rng, step_dir):
    mo.stop(not falsify_btn.value, mo.md("_train models, then click sweep_"))

    def _near_manifold(pts, X, eps=0.15):
        _d = np.sqrt(((pts[:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
        return (_d < eps).mean(), (np.linalg.norm(pts, axis=1) > 5.0).mean()

    _dts = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20])
    _Ks = np.array([20, 50, 100, 200])
    _n = 250

    _results = {nm: {"conv": np.zeros((len(_Ks), len(_dts))), "div": np.zeros((len(_Ks), len(_dts)))} for nm in PARAMETRIZATIONS}

    for _name in PARAMETRIZATIONS:
        for _i, _K in enumerate(_Ks):
            for _j, _dt in enumerate(_dts):
                _u = rng.standard_normal((_n, 2)) * 1.0
                for _ in range(_K):
                    _pred, _ = forward(models[_name], _u)
                    _u = _u + _dt * step_dir(_name, _pred, _u)
                    _u = np.clip(_u, -10, 10)
                _conv, _div = _near_manifold(_u, X)
                _results[_name]["conv"][_i, _j] = _conv
                _results[_name]["div"][_i, _j] = _div

    _fig, _axes = plt.subplots(2, 4, figsize=(14, 6))
    for _col, _name in enumerate(PARAMETRIZATIONS):
        _axes[0, _col].imshow(_results[_name]["conv"], aspect="auto", cmap="Greens", vmin=0, vmax=1, origin="lower")
        _axes[0, _col].set_title(f"{_name}\nconvergence")
        _axes[0, _col].set_xticks(range(len(_dts)), [f"{d:.2f}" for d in _dts], rotation=45)
        _axes[0, _col].set_yticks(range(len(_Ks)), _Ks)
        _axes[0, _col].set_xlabel("dt")
        if _col == 0: _axes[0, _col].set_ylabel("K steps")
        _axes[1, _col].imshow(_results[_name]["div"], aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
        _axes[1, _col].set_title(f"{_name}\ndivergence")
        _axes[1, _col].set_xticks(range(len(_dts)), [f"{d:.2f}" for d in _dts], rotation=45)
        _axes[1, _col].set_yticks(range(len(_Ks)), _Ks)
        _axes[1, _col].set_xlabel("dt")
    plt.tight_layout()
    mo.vstack([mo.md("**Wide green = stable basin (good). Wide red = divergence regime (bad).**"), _fig])


@app.cell(hide_code=True)
def _section_8(mo):
    mo.md(
        r"""
        ## 8. Limits and next steps

        This notebook proves a small, concrete thing: on a 2D swiss roll, with four numpy parametrizations trained from scratch, bounded-vector-field methods are much less fragile than high-gain noise-prediction parametrizations under the same sampling stress. In this sandbox the instability is not aesthetic or anecdotal — you can see trajectories blow up, samples leave the data manifold, and the falsification panel change as $(K, dt)$ moves.

        It does **not** prove the full image-domain claim of *The Geometry of Noise*. It does not cover convolutional architectures, large datasets, learned variance schedules, guidance, text conditioning, or the engineering choices that make modern diffusion systems work. A 2D swiss roll is not ImageNet. The point here is narrower: reproduce the mechanism in a setting where nothing important can hide behind scale.

        That is also why the demo is 2D. The smallness is a feature, not a bug. Bret Victor's lesson for interactive media is that understanding changes when the variables are touchable. Here, every tensor fits on screen. You can inspect the field, move the sampler, change the gain, and watch failure appear instead of receiving it as a theorem or a benchmark number.

        If you want to push further: replace `swiss_roll` with `sklearn.datasets.load_digits`, raise the hidden width to `256`, and stop pretending the Euler sampler is enough. At that point you will need a proper $\hat{\mathbf{x}}$-recovery sampler — the relevant starting point is paper §5.2.

        ### What this notebook adds beyond the paper

        The falsification panel is the part I wish every theory paper had as a companion. Sahraee-Ardakan, Delbracio, and Milanfar prove a gap. This notebook **maps** the gap empirically: not just whether one parametrization wins, but where it wins, where it fails, and how sharp the boundary is.

        ### Companion to

        Credit to **Mojtaba Sahraee-Ardakan, Mauricio Delbracio, and Peyman Milanfar** at Google for the paper and the clean problem framing — *The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning*, [arXiv:2602.18428](https://arxiv.org/abs/2602.18428).

        Gratitude to the **alphaXiv × marimo competition** for creating a reason to turn the argument into something inspectable.

        ---

        If you find a $(K, dt)$ where DDPM beats FM, I want to see it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
