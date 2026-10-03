"""Generates the graphs used in derivation.ipynb and saves them under graphs/.

Run `python utils/make_graphs.py` after changing any of the plots; the notebook only
embeds the saved images."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_convexity_examples():
    """Plots a strictly convex and a strictly concave function, each with a
    chord between two points, to show how the chord sits relative to the curve."""
    x = np.linspace(-2, 2, 200)
    convex_y = x ** 2
    concave_y = -x ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y, title, note in [
        (axes[0], convex_y, r"Strictly convex: $f(x) = x^2$", "chord lies on/above the curve"),
        (axes[1], concave_y, r"Strictly concave: $f(x) = -x^2$", "chord lies on/below the curve"),
    ]:
        ax.plot(x, y, color="tab:blue", label="$f(x)$")
        x1, x2 = -1.5, 1.5
        y1, y2 = np.interp([x1, x2], x, y)
        ax.plot([x1, x2], [y1, y2], "--", color="tab:orange", label="chord")
        ax.scatter([x1, x2], [y1, y2], color="tab:orange", zorder=5)
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_jensen_general():
    """Plots the concave function ln(x) with sample points, showing
    ln(E[X]) >= E[ln(X)] — the general Jensen's inequality statement, before
    the numeric log example in the next plot."""
    x = np.array([1.0, 3.0, 4.0, 6.0, 9.0])
    phi = np.log

    mean_x = x.mean()
    phi_mean = phi(mean_x)
    mean_phi = phi(x).mean()

    x_range = np.linspace(0.5, x.max() * 1.1, 200)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(x_range, phi(x_range), color="tab:blue", label=r"$\ln(x)$ (concave)")
    plt.scatter(x, phi(x), color="tab:orange", zorder=5, label="sample points, on the curve")

    plt.scatter([mean_x], [phi_mean], color="green", zorder=6, s=90,
                label=r"$\ln(\mathbb{E}[X])$")
    plt.scatter([mean_x], [mean_phi], color="red", zorder=6, s=90,
                label=r"$\mathbb{E}[\ln(X)]$")
    plt.vlines(mean_x, mean_phi, phi_mean, color="gray", linestyle=":")

    plt.xlabel("x")
    plt.ylabel(r"$\ln(x)$")
    plt.title(r"Jensen's Inequality: $\ln(\mathbb{E}[X]) \geq \mathbb{E}[\ln(X)]$")
    plt.legend()
    plt.grid(alpha=0.3)
    return fig


def _gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _mixture_pdf(x, mu1, s1, mu2, s2, w=0.5):
    return w * _gaussian_pdf(x, mu1, s1) + (1 - w) * _gaussian_pdf(x, mu2, s2)


def plot_kl_asymmetry():
    """Plots forward KL(p||q) vs reverse KL(q||p) fits of a unimodal Gaussian
    q to a bimodal true distribution p, showing that the two directions land
    on qualitatively different approximations: forward KL is mode-covering
    (q spreads to avoid zero density wherever p > 0), reverse KL is
    mode-seeking (q locks onto a single mode of p and ignores the rest)."""
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    mu1, s1, mu2, s2 = -3.0, 1.0, 3.0, 1.0
    p = _mixture_pdf(x, mu1, s1, mu2, s2)
    p = p / (p.sum() * dx)

    def kl(a, b, eps=1e-12):
        mask = a > eps
        return np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))) * dx

    # Forward KL, min_q KL(p || q): for a Gaussian q this closed-form
    # solution is moment matching (mean/variance of p).
    mean_p = np.sum(x * p) * dx
    var_p = np.sum(((x - mean_p) ** 2) * p) * dx
    q_forward = _gaussian_pdf(x, mean_p, np.sqrt(var_p))

    # Reverse KL, min_q KL(q || p): no closed form, so search over
    # Gaussian (mu, sigma) for the minimizer.
    best = None
    for mu in np.linspace(-6, 6, 121):
        for sigma in np.linspace(0.2, 3.0, 57):
            q = _gaussian_pdf(x, mu, sigma)
            q = q / (q.sum() * dx)
            val = kl(q, p)
            if best is None or val < best[0]:
                best = (val, mu, sigma)
    _, mu_r, sigma_r = best
    q_reverse = _gaussian_pdf(x, mu_r, sigma_r)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    axes[0].fill_between(x, p, color="black", alpha=0.15, label="$p(x)$ (true)")
    axes[0].plot(x, q_forward, color="tab:green", lw=2, label="$q(x)$ (fitted)")
    axes[0].set_title(r"Forward KL: $\min_q\ \mathrm{KL}(p\|q)$" + "\n(mode-covering / zero-avoiding)", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(x, p, color="black", alpha=0.15, label="$p(x)$ (true)")
    axes[1].plot(x, q_reverse, color="tab:red", lw=2, label="$q(x)$ (fitted)")
    axes[1].set_title(r"Reverse KL: $\min_q\ \mathrm{KL}(q\|p)$" + "\n(mode-seeking / zero-forcing)", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("KL divergence is asymmetric: the fitted $q$ depends on which direction you minimize")
    plt.tight_layout()
    return fig


def plot_vae_diagram(reparameterized=False):
    """Draws the basic VAE pipeline: the encoder q_phi(z|x) maps x to the
    parameters mu(x), sigma^2(x); z is sampled from N(mu(x), sigma^2(x)); the
    decoder p_theta(x|z) maps z back to a reconstruction x_hat.

    With reparameterized=True, also shows the noise eps ~ N(0, I) feeding the
    latent via z = mu(x) + sigma(x) * eps (the reparameterization trick)."""
    from matplotlib.patches import FancyBboxPatch

    # The epsilon branch hangs below the main chain, so it needs extra height.
    y_min = 0 if reparameterized else 1.9
    fig, ax = plt.subplots(figsize=(13.5, 4.2 - y_min))
    ax.set_xlim(0, 13.9)
    ax.set_ylim(y_min, 4.2)
    ax.axis("off")

    def box(cx, cy, w, h, text, color, fontsize=11):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=color, edgecolor="black", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, lw=1.5, color="black", linestyle=ls))

    y = 2.8
    box(0.8, y, 1.1, 0.9, "Input\n$x$", "#e8e8e8")
    box(3.0, y, 2.0, 1.1, "Encoder\n$q_\\phi(z \\mid x)$", "#cfe2f3")
    box(5.6, y, 1.6, 1.1, "$\\mu(x)$\n$\\sigma^2(x)$", "#fff2cc")
    box(8.2, y, 2.4, 1.1, "Latent $z$\n$z \\sim \\mathcal{N}(\\mu(x), \\sigma^2(x))$", "#d9ead3", fontsize=10)
    box(10.8, y, 1.8, 1.1, "Decoder\n$p_\\theta(x \\mid z)$", "#f4cccc")
    box(13.1, y, 1.4, 0.9, "Reconstruction\n$\\hat{x}$", "#e8e8e8", fontsize=10)

    arrow(1.35, y, 2.0, y)
    arrow(4.0, y, 4.8, y)
    arrow(6.4, y, 7.0, y)
    arrow(9.4, y, 9.9, y)
    arrow(11.7, y, 12.4, y)

    if reparameterized:
        box(8.2, 0.9, 1.9, 0.6, "$\\epsilon \\sim \\mathcal{N}(0, I)$", "#f3f3f3", fontsize=10)
        arrow(8.2, 1.2, 8.2, 2.25, ls="--")
        ax.text(8.35, 1.72, "$z = \\mu(x) + \\sigma(x) \\odot \\epsilon$", fontsize=10, va="center")

    ax.text(3.0, 3.75, "inference (encode)", ha="center", fontsize=9, color="gray")
    ax.text(10.8, 3.75, "generation (decode)", ha="center", fontsize=9, color="gray")

    if reparameterized:
        plt.title("VAE with the reparameterization trick: the randomness comes from $\\epsilon$")
    else:
        plt.title("Basic VAE: encode $x$ to a distribution, sample $z$, decode to $\\hat{x}$")
    plt.tight_layout()
    return fig


def plot_jensen_log(x, avg_x, log_avg, avg_log):
    """Plots log(x) alongside the chord through the data's endpoints, marking
    log(average(x)) vs average(log(x))."""
    x_range = np.linspace(0.5, x.max() * 1.05, 200)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(x_range, np.log(x_range), color="tab:blue", label=r"$\log(x)$ (concave function)")

    plt.scatter(x, np.log(x), color="tab:orange", zorder=5, label="example data, on the curve")
    plt.plot([x.min(), x.max()], [np.log(x.min()), np.log(x.max())],
             "--", color="tab:orange", label="chord (linear interpolation)")

    plt.scatter([avg_x], [log_avg], color="green", zorder=6, s=90,
                label=r"$\log(\mathrm{average}(x))$")
    plt.scatter([avg_x], [avg_log], color="red", zorder=6, s=90,
                label=r"$\mathrm{average}(\log(x))$")
    plt.vlines(avg_x, avg_log, log_avg, color="gray", linestyle=":")

    plt.xlabel("x")
    plt.ylabel("log(x)")
    plt.title("Jensen's Inequality: log(average) >= average(log)")
    plt.legend()
    plt.grid(alpha=0.3)
    return fig


# File name (without extension) under graphs/ -> function building the figure.
GRAPHS = {
    "convexity_examples": plot_convexity_examples,
    "jensen_general": plot_jensen_general,
    "kl_asymmetry": plot_kl_asymmetry,
    "vae_diagram": plot_vae_diagram,
    "vae_diagram_reparameterized": lambda: plot_vae_diagram(reparameterized=True),
}


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "graphs" # graphs/ sits at the repo root
    out_dir.mkdir(exist_ok=True)
    for name, plot in GRAPHS.items():
        fig = plot()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path.relative_to(out_dir.parent)}")
