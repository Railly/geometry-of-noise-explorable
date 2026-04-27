import importlib.util
import pathlib
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


K = 100
DT = 0.04
SEED = 42
N_POINTS = 200
TAIL = 10
N_TAILS = 30


def timeout_handler(signum, frame):
    pathlib.Path("/tmp/gif-status.txt").write_text(
        "Aborted: GIF generation exceeded 5 minutes wallclock.\n"
    )
    raise TimeoutError("GIF generation exceeded 5 minutes")


def load_killer_module(root):
    path = root / "scripts" / "make-killer-figure.py"
    spec = importlib.util.spec_from_file_location("killer_figure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_paths(module, name, model, start, k=K, dt=DT):
    u = start.copy()
    positions = [u.copy()]
    for _ in range(k):
        pred, _ = module.forward(model, u)
        u = u + dt * module.step_dir(name, pred, u)
        u = np.clip(u, -10, 10)
        positions.append(u.copy())
    return np.array(positions)


def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / "assets" / "kill-shot.gif"
    root.joinpath("assets").mkdir(exist_ok=True)
    module = load_killer_module(root)

    X = module.make_swiss_roll(1500, 0.05)
    seeds = {"DDPM": 934, "FM": 672}
    models = {
        name: module.train_one(name, X, steps=2500, seed=seeds[name])
        for name in ("DDPM", "FM")
    }

    g = np.random.default_rng(SEED)
    start = g.standard_normal((N_POINTS, 2))
    tail_idx = g.choice(N_POINTS, size=N_TAILS, replace=False)
    paths = {
        "FM": sample_paths(module, "FM", models["FM"], start),
        "DDPM": sample_paths(module, "DDPM", models["DDPM"], start),
    }

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.8), dpi=80)
    fig.suptitle("FM (left) vs DDPM (right) — autonomous sampling on swiss roll", fontsize=10)

    panels = {}
    for ax, name in zip(axes, ("FM", "DDPM")):
        ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.3, c="#0a84ff", edgecolors="none")
        lines = []
        for _ in range(N_TAILS * TAIL):
            line, = ax.plot([], [], lw=0.45, c="#777777", alpha=0.0)
            lines.append(line)
        dots = ax.scatter([], [], s=7, c="#ff3b30", alpha=0.9, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.12)
        panels[name] = (ax, lines, dots)

    def update(frame):
        step = frame + 1
        artists = []
        for name in ("FM", "DDPM"):
            ax, lines, dots = panels[name]
            pts = paths[name]
            dots.set_offsets(pts[step])
            ax.set_title(f"k={step}/{K}, dt={DT:.2f}", fontsize=9)
            pos = 0
            for idx in tail_idx:
                for lag in range(TAIL):
                    line = lines[pos]
                    pos += 1
                    end = step - lag
                    start_step = end - 1
                    if start_step < 0:
                        line.set_data([], [])
                        line.set_alpha(0.0)
                    else:
                        seg = pts[start_step : end + 1, idx]
                        line.set_data(seg[:, 0], seg[:, 1])
                        line.set_alpha(0.05 + 0.22 * (TAIL - lag) / TAIL)
                    artists.append(line)
            artists.append(dots)
            artists.append(ax.title)
        return artists

    anim = FuncAnimation(fig, update, frames=K, interval=40, blit=False)
    anim.save(out, writer=PillowWriter(fps=25), dpi=80)
    plt.close(fig)
    signal.alarm(0)
    print(out)


if __name__ == "__main__":
    main()
