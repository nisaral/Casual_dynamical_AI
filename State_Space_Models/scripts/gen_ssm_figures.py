"""Generate educational figures for SSM learning notes."""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def fig01():
    t = np.linspace(0, 8, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for a, label in [
        (0.3, r"slow forget $\lambda=-0.3$"),
        (1.0, r"medium $\lambda=-1$"),
        (3.0, r"fast forget $\lambda=-3$"),
    ]:
        ax.plot(t, np.exp(-a * t), lw=2.2, label=label)
    ax.set_xlabel("lag t")
    ax.set_ylabel(r"impulse response $k(t)$")
    ax.set_title("Continuous SSM memory: how a past spike fades")
    ax.legend()
    save(fig, "fig01_impulse_response.png")


def fig02():
    t = np.linspace(0, 8, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    k = np.exp(-0.25 * t) * np.cos(2.5 * t)
    ax.plot(t, k, lw=2.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag t")
    ax.set_ylabel(r"$k(t)=e^{\alpha t}\cos(\omega t)$")
    ax.set_title("Complex eigenvalues: decaying oscillation (ringing memory)")
    save(fig, "fig02_oscillatory_kernel.png")


def fig03():
    a, Delta = 1.0, 0.4
    t_c = np.linspace(0, 6, 500)
    k_c = np.exp(-a * t_c)
    ks = np.arange(0, int(6 / Delta) + 1)
    k_d = np.exp(-a * Delta * ks)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t_c, k_c, lw=2, label=r"continuous $e^{-at}$")
    ax.stem(ks * Delta, k_d, linefmt="C1-", markerfmt="C1o", basefmt=" ",
            label=rf"ZOH samples $\Delta={Delta}$")
    ax.set_xlabel("time")
    ax.set_ylabel("memory of a unit spike")
    ax.set_title("Discretization samples the continuous fade at token steps")
    ax.legend()
    save(fig, "fig03_discretization.png")


def fig04():
    L = 200
    m = np.arange(L)
    lams = -np.array([0.02, 0.08, 0.25, 1.0, 3.0])
    weights = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    modes = np.exp(np.outer(m, lams)) * weights
    K = modes.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i in range(len(lams)):
        ax.plot(m, modes[:, i], lw=1.3, alpha=0.85, label=rf"mode $\lambda={lams[i]}$")
    ax.plot(m, K, "k", lw=2.6, label=r"sum = full kernel $K_m$")
    ax.set_xlabel("lag m (tokens)")
    ax.set_ylabel("weight")
    ax.set_title("Diagonal SSM kernel = sum of geometric modes")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, L)
    save(fig, "fig04_multiscale_kernel.png")


def zoh_diag(lam, B, delta):
    Abar = np.exp(lam * delta)
    Bbar = np.where(
        np.abs(lam) < 1e-8,
        B * delta,
        B * (np.expm1(lam * delta) / lam),
    )
    return Abar, Bbar


def fig05():
    rng = np.random.default_rng(0)
    N, L, delta = 8, 256, 0.05
    lam = -np.linspace(0.1, 2.0, N)
    B = rng.normal(size=N)
    C = rng.normal(size=N)
    Abar, Bbar = zoh_diag(lam, B, delta)
    u = rng.normal(size=L)
    x = np.zeros(N)
    ys = []
    for uk in u:
        x = Abar * x + Bbar * uk
        ys.append(C @ x)
    y_scan = np.array(ys)
    K = np.zeros(L)
    v = Bbar.copy()
    for m in range(L):
        K[m] = C @ v
        v = Abar * v
    n = 1
    while n < 2 * L:
        n *= 2
    y_fft = np.fft.irfft(np.fft.rfft(K, n) * np.fft.rfft(u, n), n)[:L]
    err = np.abs(y_scan - y_fft)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2))
    axes[0].plot(y_scan[:80], lw=2, label="recurrent scan")
    axes[0].plot(y_fft[:80], "--", lw=1.8, label="FFT convolution")
    axes[0].set_title("LTI identity: scan output equals FFT(K)*u")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[1].semilogy(err, color="C3", lw=1.5)
    axes[1].set_xlabel("time index k")
    axes[1].set_ylabel("|scan - fft|")
    axes[1].set_title(f"Absolute error (max={err.max():.2e})")
    save(fig, "fig05_scan_vs_fft.png")


def fig06():
    rng = np.random.default_rng(1)
    L = 120
    t = np.arange(L)
    u = 0.15 * rng.normal(size=L)
    spike_pos = [20, 55, 90]
    for p in spike_pos:
        u[p] += 3.0
    lam = -np.array([0.3, 0.8, 1.5])
    B = np.ones(3)
    C = np.array([0.5, 0.3, 0.2])
    Abar, Bbar = zoh_diag(lam, B, 0.2)
    x = np.zeros(3)
    y_lti = []
    for uk in u:
        x = Abar * x + Bbar * uk
        y_lti.append(C @ x)
    y_lti = np.array(y_lti)
    x = np.zeros(3)
    y_sel = []
    for uk in u:
        delta_k = 0.05 + 0.6 * (abs(uk) > 1.0)
        Abar_k = np.exp(lam * delta_k)
        Bbar_k = np.where(
            np.abs(lam) < 1e-8,
            B * delta_k,
            B * (np.expm1(lam * delta_k) / lam),
        )
        x = Abar_k * x + Bbar_k * uk
        y_sel.append(C @ x)
    y_sel = np.array(y_sel)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    axes[0].plot(t, u, color="0.3", lw=1.2)
    axes[0].set_ylabel("input u")
    axes[0].set_title("Selective copying intuition: spikes in noise")
    for p in spike_pos:
        axes[0].axvline(p, color="C1", alpha=0.3, lw=2)
    axes[1].plot(t, y_lti, color="C0", lw=2)
    axes[1].set_ylabel("LTI y")
    axes[1].set_title("Fixed filter: spikes blurred into noise")
    axes[2].plot(t, y_sel, color="C2", lw=2)
    axes[2].set_ylabel("selective y")
    axes[2].set_xlabel("token index")
    axes[2].set_title(r"Selective $\Delta(u)$: reacts hard at spikes")
    save(fig, "fig06_selective_vs_lti.png")


def fig07():
    Lvals = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
    attn = Lvals ** 2
    ssm = Lvals * 64
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(Lvals, attn / attn[0], "o-", lw=2.2, label=r"Attention $\sim L^2$")
    ax.loglog(Lvals, ssm / ssm[0], "s-", lw=2.2, label=r"SSM scan $\sim L\cdot N$")
    ax.set_xlabel("sequence length L")
    ax.set_ylabel("relative compute (vs L=512)")
    ax.set_title("Long context: quadratic vs linear growth")
    ax.legend()
    save(fig, "fig07_complexity.png")


def fig08():
    L = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 1024
    layers, d = 32, 4096
    kv = 2 * layers * L * d * 2
    ssm_mem = np.full_like(L, layers * d * 16 * 2, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(L / 1024, kv / (1024 ** 3), "o-", lw=2.2, label="Transformer KV-cache (schematic FP16)")
    ax.plot(L / 1024, ssm_mem / (1024 ** 3), "s-", lw=2.2, label="SSM state N=16 (schematic FP16)")
    ax.set_xlabel("context length (thousands of tokens)")
    ax.set_ylabel("memory (GB, schematic)")
    ax.set_title("Decode memory: KV grows with L; SSM state does not")
    ax.legend()
    save(fig, "fig08_memory_growth.png")


def fig09():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (1, 4.5, 2.2, 1.0, "Registers\n(fastest, tiny)", "#d4edda"),
        (4, 4.5, 2.4, 1.0, "SRAM / Shared\n(on-SM scratch)", "#cce5ff"),
        (7, 4.5, 2.4, 1.0, "L2 cache", "#e2e3e5"),
        (2.5, 2.2, 5, 1.3, "HBM (GPU RAM)\nLARGE but SLOW", "#f8d7da"),
    ]
    for x, y, w, h, text, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="k", lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    ax.text(5, 3.7, "fuse kernels so state stays high in the stack", ha="center", fontsize=9)
    ax.set_title("GPU memory hierarchy (why fused Mamba scan exists)", pad=12)
    save(fig, "fig09_gpu_hierarchy.png")


def fig10():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.8)
    ax.axis("off")
    for i in range(8):
        ax.add_patch(plt.Circle((i, 0), 0.28, facecolor="#cce5ff", edgecolor="k"))
        ax.text(i, 0, f"P{i+1}", ha="center", va="center", fontsize=8)
    for i, x in enumerate([0.5, 2.5, 4.5, 6.5]):
        ax.add_patch(plt.Circle((x, 1.5), 0.28, facecolor="#d4edda", edgecolor="k"))
        ax.text(x, 1.5, "+", ha="center", va="center")
        ax.plot([2 * i, x], [0.28, 1.22], "k-", lw=1)
        ax.plot([2 * i + 1, x], [0.28, 1.22], "k-", lw=1)
    for x, kids in [(1.5, [0.5, 2.5]), (5.5, [4.5, 6.5])]:
        ax.add_patch(plt.Circle((x, 3.0), 0.28, facecolor="#fff3cd", edgecolor="k"))
        ax.text(x, 3.0, "+", ha="center", va="center")
        for k in kids:
            ax.plot([k, x], [1.78, 2.72], "k-", lw=1)
    ax.add_patch(plt.Circle((3.5, 4.2), 0.3, facecolor="#f8d7da", edgecolor="k"))
    ax.text(3.5, 4.2, "+", ha="center", va="center")
    ax.plot([1.5, 3.5], [3.28, 3.92], "k-", lw=1)
    ax.plot([5.5, 3.5], [3.28, 3.92], "k-", lw=1)
    ax.text(7.2, 3.5, "depth ~ log L\nwork ~ L", fontsize=10)
    ax.set_title("Parallel associative scan tree (Blelloch prefix)", pad=8)
    save(fig, "fig10_parallel_scan.png")


def hippo_legt(N):
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def fig11():
    ev = eigvals(hippo_legt(32))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(ev.real, ev.imag, c="C0", s=40, edgecolors="k", linewidths=0.4)
    ax.axvline(0, color="k", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("HiPPO-LegT eigenvalues (N=32): stable left half-plane")
    ax.set_aspect("equal", adjustable="datalim")
    save(fig, "fig11_hippo_eigs.png")


def fig12():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, t, c):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="k", lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)

    box(0.2, 1.5, 1.5, 1.0, "Input\nu", "#e2e3e5")
    box(2.0, 1.5, 1.6, 1.0, "Linear\nexpand", "#cce5ff")
    box(3.9, 1.5, 1.5, 1.0, "Conv1d\nlocal", "#cce5ff")
    box(5.7, 1.5, 1.8, 1.0, "Selective\nSSM", "#d4edda")
    box(7.8, 1.5, 1.4, 1.0, "Gate\nSiLU", "#fff3cd")
    box(9.5, 1.5, 1.5, 1.0, "Project\n+ resid", "#f8d7da")
    for x0, x1 in [(1.7, 2.0), (3.6, 3.9), (5.4, 5.7), (7.5, 7.8), (9.2, 9.5)]:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(6.0, 0.5, "One Mamba block (stacked many times)", ha="center", fontsize=11)
    ax.set_title("Real architecture path inside a Mamba layer", pad=8)
    save(fig, "fig12_mamba_block.png")


def fig13():
    a_vals = np.linspace(0.2, 3.0, 80)
    d_vals = np.linspace(0.02, 0.5, 80)
    A, D = np.meshgrid(a_vals, d_vals)
    H = np.log(2) / (A * D)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.pcolormesh(a_vals, d_vals, H, shading="auto", cmap="viridis")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("half-life (steps)")
    ax.set_xlabel(r"decay rate a (A=-a)")
    ax.set_ylabel(r"step size Delta")
    ax.set_title(r"Scalar ZOH half-life m=ln2/(a Delta)")
    save(fig, "fig13_halflife.png")


def fig14():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    items = [
        (0.3, 4.2, 2.8, 1.3, "Long document\n/ code repo", "#cce5ff"),
        (3.5, 4.2, 3.0, 1.3, "Hybrid or Mamba\nLM (GPU kernels)", "#d4edda"),
        (6.9, 4.2, 2.8, 1.3, "API / IDE\ncompletion", "#fff3cd"),
        (0.3, 1.5, 2.8, 1.3, "Prefill\n(prompt encode)", "#e2e3e5"),
        (3.5, 1.5, 3.0, 1.3, "State cache\nO(N) not O(L)", "#d4edda"),
        (6.9, 1.5, 2.8, 1.3, "Decode loop\nfast tokens/s", "#f8d7da"),
    ]
    for x, y, w, h, t, c in items:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="k", lw=1.3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    ax.set_title("Industry path: long context in, efficient SSM/hybrid out", pad=10)
    save(fig, "fig14_industry_pipeline.png")


if __name__ == "__main__":
    for fn in [
        fig01, fig02, fig03, fig04, fig05, fig06, fig07,
        fig08, fig09, fig10, fig11, fig12, fig13, fig14,
    ]:
        fn()
    print("done")
