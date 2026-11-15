"""Generate the result plots used in the defence slides.

All numbers are read from scripts/data/*.csv, which are transcribed from the
thesis tables (each CSV names its source table in its first line).
The margin-time figures are computed from a simulated dataset.

Usage (from the repo root):
    python scripts/make_plots.py
Writes figures/plot_*.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "data"
OUT = ROOT / "figures"

# ---------------------------------------------------------------- style
# Validated reference hues (dataviz skill palette, light mode) + neutral gray.
BLUE = "#2a78d6"
ORANGE = "#eb6834"   # proposed / highlighted method
AQUA = "#1baf7a"
GRAY = "#9a9a9a"
INK = "#1f1f1f"
MUTED = "#5a5a5a"
GRID = "#e6e6e6"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def read(name):
    return pd.read_csv(DATA / name, comment="#")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def hgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ MENSA results
def mensa_results():
    df = read("mensa_results.csv")
    df["dcal"] = 100 * df["dcal_pass"] / df["dcal_total"]
    datasets = ["MIMIC-IV", "Rotterdam", "PRO-ACT", "EBMT"]
    ks = {"MIMIC-IV": 3, "Rotterdam": 2, "PRO-ACT": 4, "EBMT": 5}
    # Only the two C-indices (both: higher is better) -- calibration is in the text
    metrics = [  # column, sd column, title, higher is better
        ("ci_g", "ci_g_sd", "Global C-index", True),
        ("ci_l", "ci_l_sd", "Local C-index", True),
    ]
    style = {
        "MENSA": dict(c=ORANGE, m="D", s=46, z=5),
        "DeepHit": dict(c=BLUE, m="s", s=26, z=4),
        "Hierarchical": dict(c=AQUA, m="^", s=30, z=4),
    }
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.2, 2.6),
                             gridspec_kw=dict(hspace=1.1, wspace=0.15))
    for r, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        for c, (col, sdcol, title, up) in enumerate(metrics):
            ax = axes[r, c]
            vals = sub[col].values
            core = sub[~sub.model.isin(["DeepHit", "Hierarchical"])][col].values
            lo, hi = core.min(), core.max()
            span = max(hi - lo, 1e-6)
            if col == "dcal":
                xmin, xmax = -5, 105
            else:
                pad = 0.35 * span
                xmin, xmax = lo - pad, hi + pad
                # extend to include DeepHit / Hierarchical when they are close
                for v in vals:
                    if xmin - 0.6 * span < v < xmin:
                        xmin = v - 0.1 * span
                    if xmax < v < xmax + 0.6 * span:
                        xmax = v + 0.1 * span
            # baselines (gray) first, highlighted models on top
            sub_sorted = sorted(sub.itertuples(), key=lambda r: r.model in style)
            for row in sub_sorted:
                st = style.get(row.model, dict(c=GRAY, m="o", s=16, z=3))
                v = getattr(row, col)
                ms = np.sqrt(st["s"])
                worse_dir = (v < xmin) if up else (v > xmax)
                if v < xmin or v > xmax:
                    # off-scale (always on the "worse" side here): marker pinned
                    # to the axis edge, pointing outwards
                    x = xmin if v < xmin else xmax
                    pointing_left = (x == xmin) != (not up)
                    ax.plot([x], [0], marker="<" if pointing_left else ">", ms=ms,
                            color=st["c"], mec="none", ls="",
                            zorder=st["z"], clip_on=False)
                    continue
                ax.plot([v], [0], marker=st["m"], ms=ms, color=st["c"], mec="none",
                        mew=0.5, ls="", zorder=st["z"],
                        alpha=1.0 if row.model in style else 0.8)
            ax.set_xlim(xmin, xmax)
            if not up:
                ax.invert_xaxis()   # so "better" is always to the right
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.axhline(0, color=GRID, lw=0.8, zorder=1)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.tick_params(axis="x", length=2, pad=1.5, labelsize=7)
            if r == 0:
                ax.set_title(title, pad=10, fontweight="bold")
            if c == 0:
                ax.text(-0.05, 0.15, f"{ds}\n($K$={ks[ds]})", transform=ax.transAxes,
                        ha="right", va="center", fontsize=7.5)
    fig.text(0.53, -0.03, "higher is better  →", ha="center", fontsize=7.5, color=MUTED)
    handles = [
        plt.Line2D([], [], marker="D", ls="", color=ORANGE, mec="none", ms=6, label="MENSA (ours)"),
        plt.Line2D([], [], marker="^", ls="", color=AQUA, mec="none", ms=6, label="Hierarchical (multi-event)"),
        plt.Line2D([], [], marker="s", ls="", color=BLUE, mec="none", ms=5.5, label="DeepHit (competing risks)"),
        plt.Line2D([], [], marker="o", ls="", color=GRAY, mec="none", ms=5, label="8 other baselines"),
        plt.Line2D([], [], marker="<", ls="", color=MUTED, mec="none", ms=5, label="off-scale (worse)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.13), handletextpad=0.3, columnspacing=1.0)
    save(fig, "plot_mensa_results.pdf")


def mensa_ablation():
    """Change in local C-index when removing a component (full minus ablated),
    with the run-to-run SD of the full model as a whisker."""
    df = read("mensa_ablation.csv")
    rows = [("joint", "Rotterdam", "Joint training", "Rotterdam"),
            ("joint", "PRO-ACT", "Joint training", "PRO-ACT"),
            ("trajectory", "Rotterdam", "Trajectory loss", "Rotterdam"),
            ("trajectory", "EBMT", "Trajectory loss", "EBMT")]
    fig, ax = plt.subplots(figsize=(2.7, 2.0))
    ys = np.arange(len(rows))
    for y, (study, ds, comp, lab) in zip(ys, rows):
        sub = df[(df.study == study) & (df.dataset == ds)].reset_index(drop=True)
        full, abl = sub.loc[0], sub.loc[1]
        delta = full.ci_l - abl.ci_l
        ax.plot([delta - full.ci_l_sd, delta + full.ci_l_sd], [y, y], color=GRAY, lw=1.4,
                solid_capstyle="round", zorder=2)
        ax.plot([0, delta], [y, y], color=ORANGE, lw=3, solid_capstyle="butt", zorder=3)
        ax.plot([delta], [y], "o", color=ORANGE, ms=5, zorder=4)
        ax.annotate(f"+{delta:.2f}", (delta, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys, [f"{r[2]}\n{r[3]}" for r in rows])
    ax.set_xlim(-2.2, 4.2)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlabel("Gain in local C-index (points)")
    hgrid(ax)
    handles = [plt.Line2D([], [], color=ORANGE, lw=3, label="gain from component"),
               plt.Line2D([], [], color=GRAY, lw=1.4, label="± 1 SD over 10 runs")]
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.6, 1.14))
    save(fig, "plot_mensa_ablation.pdf")


# ------------------------------------------------------------ BaySurv
def baysurv():
    df = read("baysurv_results.csv")
    params = read("baysurv_params.csv")
    models = ["Baseline (MLP)", "VI", "MCD (p=0.5)", "SNGP"]
    labels = {"Baseline (MLP)": "MLP baseline", "VI": "VI", "MCD (p=0.5)": "MC dropout", "SNGP": "SNGP"}
    st = {"Baseline (MLP)": (GRAY, "o"), "VI": (BLUE, "s"),
          "MCD (p=0.5)": (ORANGE, "D"), "SNGP": (AQUA, "^")}
    datasets = ["METABRIC", "SEER", "SUPPORT", "MIMIC-IV"]

    fig, axes = plt.subplots(1, 3, figsize=(4.4, 2.0),
                             gridspec_kw=dict(width_ratios=[1.4, 1, 1], wspace=0.7))
    # (a) C-index per dataset
    ax = axes[0]
    offs = np.linspace(-0.22, 0.22, len(models))
    for j, m in enumerate(models):
        c, mk = st[m]
        for i, ds in enumerate(datasets):
            v = df[(df.dataset == ds) & (df.model == m)].ci_td.item()
            ax.scatter(v, i + offs[j], color=c, marker=mk, s=26, edgecolor="none",
                       linewidth=0.5, zorder=3, label=labels[m] if i == 0 else None)
    ax.set_yticks(range(len(datasets)), datasets)
    ax.invert_yaxis()
    ax.set_xlabel("C-index (Antolini)")
    ax.set_title("Discrimination", fontweight="bold")
    hgrid(ax)

    # (b) parameters on MIMIC-IV
    ax = axes[1]
    pmap = dict(zip(params.model, params.params))
    pm = ["Baseline (MLP)", "VI", "MCD", "SNGP"]
    cols = [st["Baseline (MLP)"][0], st["VI"][0], st["MCD (p=0.5)"][0], st["SNGP"][0]]
    y = np.arange(len(pm))
    ax.barh(y, [pmap[m] for m in pm], color=cols, height=0.6)
    for yi, m in zip(y, pm):
        ax.text(pmap[m] + 120, yi, f"{pmap[m]:,}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y, ["MLP", "VI", "MCD", "SNGP"])
    ax.invert_yaxis()
    ax.set_xlim(0, 7400)
    ax.set_xticks([0, 3000, 6000])
    ax.set_xlabel("Parameters")
    ax.set_title("Cost (MIMIC-IV)", fontweight="bold")
    hgrid(ax)

    # (c) number of D-calibrated datasets
    ax = axes[2]
    counts = [(df[(df.model == m)].dcal_p > 0.05).sum() for m in models]
    ax.barh(y, counts, color=[st[m][0] for m in models], height=0.6)
    for yi, n in zip(y, counts):
        ax.text(n + 0.08, yi, f"{n}/4", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y, ["MLP", "VI", "MCD", "SNGP"])
    ax.invert_yaxis()
    ax.set_xlim(0, 4.9)
    ax.set_xticks([0, 2, 4])
    ax.set_xlabel("# D-calibrated")
    ax.set_title("Calibration", fontweight="bold")
    hgrid(ax)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.12), handletextpad=0.2, columnspacing=0.8)
    save(fig, "plot_baysurv.pdf")


# ------------------------------------------------------------ TMJ conformal
def tmj():
    df = read("tmj_coverage.csv")
    fig, ax = plt.subplots(figsize=(2.8, 2.1))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df.tmj0, w, color=GRAY, label="No involvement (TMJ0)")
    ax.bar(x + w / 2, df.tmj1, w, color=ORANGE, label="Involvement (TMJ1)")
    ax.axhline(0.90, color=INK, lw=1, ls="--", label="Target coverage (90%)")
    for xi, v in zip(x, df.tmj1):
        ax.text(xi + w / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(x, df.label)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Empirical coverage")
    # group labels
    for lo, hi, name in [(0, 0, "IID"), (1, 3, "Temporal"), (4, 5, "Lagged")]:
        ax.annotate("", xy=(lo - 0.4, -0.2), xytext=(hi + 0.4, -0.2), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.text((lo + hi) / 2, -0.29, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.36), ncol=1, frameon=False, fontsize=6.5)
    save(fig, "plot_tmj_coverage.pdf")


# ------------------------------------------------------------ Copula Brier
def copula():
    err = read("copula_error.csv")
    rank = read("copula_ranking.csv").set_index("metric")
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.1),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.75))
    ax = axes[0]
    series = [("ipcw_coxph", "IPCW (CoxPH)", GRAY),
              ("dep_cg", "IBS-Dep", BLUE),
              ("dep_cg_uw", "IBS-Dep (UW)", ORANGE)]
    y = np.arange(len(err))
    h = 0.26
    # shown as a reduction (positive = less error than IPCW with a KM censoring model)
    for k, (col, lab, c) in enumerate(series):
        ax.barh(y + (k - 1) * h, -err[col], h * 0.92, color=c, label=lab)
    for yi, v in zip(y, err.dep_cg_uw):
        ax.text(-v + 0.4, yi + h, f"{-v:.0f}%", va="center", ha="left", fontsize=6.5, color=MUTED)
    # a negative reduction (= more error than IPCW (KM)) is drawn left of zero and labelled
    for k, (col, _, _) in enumerate(series):
        for yi, v in zip(y, err[col]):
            if -v < 0:
                ax.text(0.4, yi + (k - 1) * h, f"−{v:.1f}%", va="center", ha="left",
                        fontsize=6.5, color=MUTED)
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(y, err.strategy)
    ax.invert_yaxis()
    ax.set_xlim(-3, 22)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("Reduction in IBS estimation error\nvs. IPCW (KM)  [%]")
    ax.set_title("Error reduction (higher = better)", fontweight="bold")
    hgrid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.12))

    ax = axes[1]
    order = ["IPCW (KM)", "IPCW (CoxPH)", "Dep (CG)", "Dep (CG) UW"]
    names = ["IPCW (KM)", "IPCW (CoxPH)", "IBS-Dep", "IBS-Dep (UW)"]
    cols = [GRAY, GRAY, BLUE, ORANGE]
    share = [100 * rank.loc[m].sum() / (40 * rank.shape[1]) for m in order]
    yy = np.arange(len(order))
    ax.barh(yy, share, color=cols, height=0.6)
    for yi, s in zip(yy, share):
        ax.text(s + 1.5, yi, f"{s:.0f}%", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Runs recovering $\\geq$2 of the\noracle top-3 learners  [%]")
    ax.set_title("Model ranking", fontweight="bold")
    hgrid(ax)
    save(fig, "plot_copula_results.pdf")


# ------------------------------------------------------------ Margin time
def km_cg(t, d, theta):
    """KM (theta=0) or Clayton Copula-Graphic estimator (Rivest & Wells, Eq. 5.1).
    Returns event times and survival estimates right after each event."""
    order = np.argsort(t)
    t, d = t[order], d[order]
    n = len(t)
    if theta == 0:
        phi = lambda u: -np.log(u)
        phi_inv = lambda s: np.exp(-s)
    else:
        phi = lambda u: (u ** (-theta) - 1.0) / theta
        phi_inv = lambda s: (1.0 + theta * s) ** (-1.0 / theta)
    s = 0.0
    times, surv = [0.0], [1.0]
    for i in range(n):
        n_at_risk = n - i
        if d[i] == 1 and n_at_risk > 1:
            s += phi((n_at_risk - 1) / n) - phi(n_at_risk / n)
            times.append(t[i])
            surv.append(phi_inv(s))
    return np.array(times), np.array(surv)


def step_eval(times, surv, x):
    idx = np.searchsorted(times, x, side="right") - 1
    return surv[np.clip(idx, 0, len(surv) - 1)]


def margin_time(times, surv, c, grid):
    """e_margin = c + int_c^inf S(t) dt / S(c), with linear tail extrapolation
    to zero beyond the last observed time (Haider et al. 2020)."""
    s = step_eval(times, surv, grid)
    t_last, s_last = times[-1], surv[-1]
    t_zero = t_last / (1.0 - s_last) if s_last < 1 else t_last  # line through (0,1) & last point
    tail = grid > t_last
    s[tail] = np.clip(1.0 - grid[tail] / t_zero, 0, None)
    m = grid >= c
    area = np.trapezoid(s[m], grid[m])
    return c + area / step_eval(times, surv, np.array([c]))[0], s


def margin_figures():
    rng = np.random.default_rng(7)
    n, tau = 400, 0.5
    theta = 2 * tau / (1 - tau)  # Clayton: tau = theta / (theta + 2)
    # Clayton copula sample via conditional inversion
    u = rng.uniform(size=n)
    w = rng.uniform(size=n)
    v = ((w ** (-theta / (1 + theta)) - 1) * u ** (-theta) + 1) ** (-1 / theta)
    # Weibull marginals (survival-scale): E = scale*(-log U)^(1/k)
    e = 6.0 * (-np.log(u)) ** (1 / 1.5)
    cens = 5.0 * (-np.log(v)) ** (1 / 1.5)
    admin = 10.0
    t = np.minimum.reduce([e, cens, np.full(n, admin)])
    d = (e <= np.minimum(cens, admin)).astype(int)

    grid = np.linspace(0, 16, 4001)
    s_true = np.exp(-(grid / 6.0) ** 1.5)
    tk, sk = km_cg(t, d, 0.0)
    tc, sc = km_cg(t, d, theta)
    c = 2.5
    m_km, s_km_full = margin_time(tk, sk, c, grid)
    m_cg, s_cg_full = margin_time(tc, sc, c, grid)
    m_true = c + np.trapezoid(s_true[grid >= c], grid[grid >= c]) / np.exp(-(c / 6.0) ** 1.5)
    print(f"margin time: KM={m_km:.2f}  CG={m_cg:.2f}  true={m_true:.2f}")

    xmax = 11.0
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(2.6, 2.3))
        if step == 1:
            main_t, main_s, main_full, m, col, lab = tk, sk, s_km_full, m_km, BLUE, "Kaplan–Meier $\\hat S_{KM}(t)$"
        else:
            main_t, main_s, main_full, m, col, lab = tc, sc, s_cg_full, m_cg, BLUE, "Copula-Graphic $\\hat S_{CG}(t)$"
            ax.plot(grid, s_true, color=INK, lw=1, ls=":", label="True $S_E(t)$", zorder=3)
            ax.step(np.append(tk, tk[-1]), np.append(sk, sk[-1]), where="post",
                    color=GRAY, lw=1.1, alpha=0.8, label="Kaplan–Meier", zorder=2)
            ax.plot([m_km], [0], marker="x", color=GRAY, ms=6, mew=1.5, clip_on=False, zorder=4)
            ax.text(m_km, -0.13, f"KM\n$\\approx${m_km:.1f}", color=MUTED, ha="center", va="top",
                    fontsize=7, transform=ax.get_xaxis_transform())
        # shaded remaining area
        mask = (grid >= c) & (grid <= xmax)
        ax.fill_between(grid[mask], 0, main_full[mask], step=None, color=col, alpha=0.13, lw=0)
        ax.step(main_t, main_s, where="post", color=col, lw=1.8, label=lab, zorder=4)
        tail = grid > main_t[-1]
        ax.plot(grid[tail], main_full[tail], color=col, lw=1, ls="--", alpha=0.6, zorder=3)
        sc_val = step_eval(main_t, main_s, np.array([c]))[0]
        # censoring time and S(c)
        ax.plot([c, c], [0, sc_val], color=ORANGE, lw=1.2, ls="--")
        ax.plot([c], [sc_val], "o", color=INK, ms=4, zorder=5)
        ax.plot([c], [0], marker="x", color=ORANGE, ms=7, mew=2, clip_on=False, zorder=6)
        ax.text(c, -0.13, "censored\nat $c$", color=ORANGE, ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())
        # margin time
        ax.plot([m, m], [0, step_eval(main_t, main_s, np.array([m]))[0] if m <= main_t[-1]
                         else np.interp(m, grid, main_full)], color=col, lw=1.2, ls="--")
        ax.plot([m], [0], marker="x", color=col, ms=7, mew=2, clip_on=False, zorder=6)
        ax.text(m, -0.13, ("copula" if step == 2 else "KM") + f"\n$\\approx${m:.1f}", color=col,
                ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Time", labelpad=18)
        ax.set_ylabel("Survival probability")
        ax.legend(loc="upper right", frameon=False, fontsize=6)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        save(fig, f"plot_margin_time_{'km' if step == 1 else 'cg'}.pdf")


# ------------------------------------------------------------ f, F, S, h, H
def survival_relations():
    """One log-normal distribution seen as f, S/F, h and H (thesis Summary 1).
    Right-skewed density with an early peak (as in the D-calibration figure);
    the hazard rises then falls -- an instantaneous risk, not a cumulative curve."""
    from scipy.stats import lognorm
    mu, sigma, ts = 0.5, 0.8, 1.5        # log-mean, log-sd, marked time t*
    dist = lognorm(s=sigma, scale=np.exp(mu))
    t = np.linspace(1e-4, 7, 801)
    f = dist.pdf(t)
    S = dist.sf(t)
    F = 1 - S
    h = f / S
    H = -np.log(S)
    Sts = float(dist.sf(ts))

    fig, axes = plt.subplots(2, 2, figsize=(2.75, 2.45),
                             gridspec_kw=dict(hspace=0.75, wspace=0.45))
    (a, b), (c, d) = axes
    # PDF: area left of t* is F(t*), area right of t* is S(t*)
    a.plot(t, f, color=INK, lw=1.4)
    a.fill_between(t[t <= ts], 0, f[t <= ts], color=GRAY, alpha=0.45, lw=0)
    a.fill_between(t[t >= ts], 0, f[t >= ts], color=BLUE, alpha=0.35, lw=0)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=GRAY, alpha=0.45, label="$F(t^*)$"),
                      Patch(color=BLUE, alpha=0.35, label="$S(t^*)$")],
             frameon=False, fontsize=6, loc="upper right", handlelength=1.0,
             borderaxespad=0.1)
    a.set_ylim(0, f.max() * 2.1)
    a.set_title("PDF $f(t)$", fontsize=7.5)
    # S and F
    b.plot(t, S, color=BLUE, lw=1.4, label="$S(t)$")
    b.plot(t, F, color=GRAY, lw=1.2, ls="--", label="$F(t)$")
    b.plot([ts], [Sts], "o", color=BLUE, ms=3.5)
    b.set_ylim(0, 1.05)
    b.legend(frameon=False, fontsize=6, loc="center right", handlelength=1.5)
    b.set_title("$S(t) = 1 - F(t)$", fontsize=7.5)
    # hazard: rises, then falls
    c.plot(t, h, color=ORANGE, lw=1.4)
    c.set_ylim(0, h.max() * 1.25)
    c.set_title("Hazard $h(t) = f/S$", fontsize=7.5)
    # cumulative hazard
    d.plot(t, H, color=AQUA, lw=1.4)
    d.set_title("$H(t) = -\\log S(t)$", fontsize=7.5)
    for ax in axes.flat:
        ax.axvline(ts, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(0, 7)
        ax.set_xticks([0, ts, 4, 6])
        ax.set_xticklabels(["0", "$t^*$", "4", "6"])
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in (c, d):
        ax.set_xlabel("Time $t$", fontsize=6.5, labelpad=1)
    save(fig, "plot_survival_relations.pdf")


# ------------------------------------------------------------ ALS (PLOS ONE)
ALS_EVENTS = ["Speech", "Swallowing", "Handwriting", "Walking", "Dyspnea"]


def grid_3_2(figsize, vgap=0.2, h=0.33, top=0.1):
    """Five axes: three on the top row, two centred on the bottom row."""
    # explicit positions so the horizontal gap is identical in both rows
    fig = plt.figure(figsize=figsize)
    w, g = 0.25, 0.08            # panel width and gap (figure fractions)
    x0 = (1 - (3 * w + 2 * g)) / 2 + 0.03
    y_top = 1 - top - h
    y_bot = y_top - vgap - h
    xs_top = [x0 + k * (w + g) for k in range(3)]
    xs_bot = [x0 + (w + g) / 2 + k * (w + g) for k in range(2)]
    axes = [fig.add_axes([x, y_top, w, h]) for x in xs_top]
    axes += [fig.add_axes([x, y_bot, w, h]) for x in xs_bot]
    return fig, axes


def als_counterfactuals():
    """Mr. Smith's five ISDs under two values of one covariate (data exported
    from the survival-als project by scripts/export_als_curves.py)."""
    curves = read("als_counterfactual_curves.csv")
    lr = read("als_counterfactual_logrank.csv")
    for feat in ["onset", "riluzole", "alsfrs", "fvc"]:
        fig, axes = grid_3_2((4.3, 2.75))
        info = lr[lr.feature == feat].set_index("event")
        for ax, ev in zip(axes, ALS_EVENTS):
            d = curves[(curves.feature == feat) & (curves.event == ev)]
            ax.step(d.time, d.s_a, where="post", color=BLUE, lw=1.4)
            ax.step(d.time, d.s_b, where="post", color=ORANGE, lw=1.4)
            ax.axhline(0.5, color=MUTED, lw=0.7, ls="--")
            p = info.loc[ev, "p_value"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(0.04, 0.06, f"log-rank {ptxt}", transform=ax.transAxes, fontsize=6.5,
                    color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, lw=0.5))
            ax.set_title(ev, fontsize=10, fontweight="bold", pad=3)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, d.time.max())
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6.5, length=2, pad=1.5)
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
        for ax in axes[3:]:
            ax.set_xlabel("Time [days]", fontsize=7, labelpad=1)
        axes[0].set_ylabel("$S(t \\mid x)$", fontsize=7.5)
        axes[3].set_ylabel("$S(t \\mid x)$", fontsize=7.5)
        la, lb = info.iloc[0].label_a, info.iloc[0].label_b
        handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label=f"Mr. Smith, {la}"),
                   plt.Line2D([], [], color=ORANGE, lw=1.6, label=f"Mr. Smith, {lb}")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.5, 1.07))
        save(fig, f"plot_als_cf_{feat}.pdf")


def als_model_vs_km():
    """mMAE (days) per model and event with 95% CI; dashed line = Kaplan-Meier."""
    from scipy import stats
    d = read("als_model_vs_km.csv")
    order = ["coxph", "rsf", "deepsurv", "mtlr", "mensa"]
    names = {"coxph": "CoxPH", "rsf": "RSF", "deepsurv": "DeepSurv", "mtlr": "MTLR", "mensa": "MENSA"}
    fig, axes = grid_3_2((4.3, 2.75), vgap=0.3, h=0.32, top=0.05)
    for k, (ax, ev) in enumerate(zip(axes, ALS_EVENTS), start=1):
        sub = d[d.EventId == k]
        km = sub.MAEKM.mean()
        for j, m in enumerate(order):
            x = sub[sub.ModelName == m].MAEM.values
            ci = stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)
            col = ORANGE if m == "mensa" else GRAY
            ax.errorbar([j], [x.mean()], yerr=[ci], fmt="o", color=col, ms=3.5,
                        elinewidth=1.1, capsize=0)
        ax.axhline(km, color=INK, lw=0.9, ls="--")
        ax.set_title(ev, fontsize=10, fontweight="bold", pad=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[m] for m in order], rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1.5)
        ax.tick_params(axis="x", length=0, pad=1)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        # keep the RSF outlier on Dyspnea from flattening the panel
        if ev == "Dyspnea":
            ax.set_ylim(120, 420)
            rsf = sub[sub.ModelName == "rsf"].MAEM.mean()
            ax.annotate(f"RSF {rsf:.0f}↑", (1, 410), ha="center", va="top", fontsize=6, color=MUTED)
    axes[0].set_ylabel("mMAE [days]", fontsize=7.5)
    axes[3].set_ylabel("mMAE [days]", fontsize=7.5)
    handles = [plt.Line2D([], [], color=INK, lw=0.9, ls="--", label="Kaplan–Meier"),
               plt.Line2D([], [], color=ORANGE, marker="o", ls="", ms=4, label="MENSA"),
               plt.Line2D([], [], color=GRAY, marker="o", ls="", ms=4, label="Other models (mean ± 95% CI)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.14))
    save(fig, "plot_als_model_vs_km.pdf")


# ------------------------------------------------------------ metric types
def metric_illustrations():
    """Re-drawn version of thesis Fig. 2.6 (Qi et al.), one figure per metric type,
    with slide-sized fonts. Patients A-D have toy survival curves."""
    from scipy.stats import norm
    t = np.linspace(0, 3, 601)
    SA = 1 - 0.35 * norm.cdf((t - 0.55) / 0.25) - 0.5 * norm.cdf((t - 2.45) / 0.4)
    SB = 1 - norm.cdf((t - 0.95) / 0.45)
    SC = 1 - norm.cdf((t - 1.6) / 0.6)
    SD = 1 - norm.cdf((t - 2.0) / 0.25)
    cols = {"A": BLUE, "B": ORANGE, "C": AQUA, "D": "#c4314b"}
    curves = {"A": SA, "B": SB, "C": SC, "D": SD}
    true = {"A": 1.5, "B": 1.8, "C": 0.8, "D": 2.15}

    def S(k, x):
        return float(np.interp(x, t, curves[k]))

    def median(k):
        c = curves[k]
        return float(t[np.argmax(c <= 0.5)])

    def base(ax, title, keys):
        for k in keys:
            ax.plot(t, curves[k], color=cols[k], lw=1.4)
            ax.plot([true[k]], [0.0], marker="*", ms=7, color=cols[k], clip_on=False, zorder=5)
        ax.set_title(title, fontsize=7.5, fontweight="bold", pad=2)
        ax.set_xlim(0, 3)
        ax.set_ylim(-0.04, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 1, 2, 3])
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    def tri(ax, x, y, c):
        ax.plot([x], [y], marker="v", ms=6, color=c, zorder=6)

    def legend(fig, keys, y=1.03):
        h = [plt.Line2D([], [], color=cols[k], lw=1.5, label=f"Patient {k}") for k in keys]
        h += [plt.Line2D([], [], marker="*", ls="", color=MUTED, ms=7, label="true time"),
              plt.Line2D([], [], marker="v", ls="", color=MUTED, ms=5, label="prediction")]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=6.3,
                   bbox_to_anchor=(0.55, y), handlelength=1.4, columnspacing=0.8)

    # ---------------- discrimination
    fig, axes = plt.subplots(3, 1, figsize=(2.1, 3.0), gridspec_kw=dict(hspace=0.8))
    a, b, c = axes
    base(a, "C-index: order of predicted times", "AB")
    a.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    for k in "AB":
        tri(a, median(k), 0.5, cols[k])
    ts = 1.65
    base(b, f"AUROC at $t^*$", "AB")
    b.axvline(ts, color=MUTED, lw=0.7, ls="--")
    for k in "AB":
        tri(b, ts, S(k, ts), cols[k])
    base(c, "Time-dependent C-index", "AB")
    c.axvline(true["A"], color=MUTED, lw=0.7, ls="--")
    for k in "AB":
        tri(c, true["A"], S(k, true["A"]), cols[k])
    c.set_xlabel("Time", fontsize=6.5, labelpad=1)
    legend(fig, "AB", 1.06)
    save(fig, "plot_metrics_discrimination.pdf")

    # ---------------- error
    fig, axes = plt.subplots(3, 1, figsize=(2.1, 3.0), gridspec_kw=dict(hspace=0.8))
    a, b, c = axes
    base(a, "MAE: predicted vs. true time", "A")
    a.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    tri(a, median("A"), 0.5, BLUE)
    a.annotate("", xy=(median("A"), 0.22), xytext=(true["A"], 0.22),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    a.text((median("A") + true["A"]) / 2, 0.27, "$|e - \\hat e|$", ha="center", fontsize=6.5)
    ts = 1.65
    base(b, "Brier score at $t^*$", "A")
    b.axvline(ts, color=MUTED, lw=0.7, ls="--")
    tri(b, ts, S("A", ts), BLUE)
    b.annotate("", xy=(ts - 0.12, S("A", ts)), xytext=(ts - 0.12, 0.0),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    b.text(ts - 0.2, S("A", ts) / 2, "$(0 - \\hat S(t^*))^2$", ha="right", va="center", fontsize=6.3)
    base(c, "Integrated Brier score", "A")
    m1 = t <= true["A"]
    c.fill_between(t[m1], SA[m1], 1, color=GRAY, alpha=0.35, lw=0)
    c.fill_between(t[~m1], 0, SA[~m1], color=GRAY, alpha=0.35, lw=0)
    c.axvline(true["A"], color=MUTED, lw=0.7, ls="--")
    c.text(0.75, 0.8, "$\\alpha$", fontsize=7)
    c.text(2.1, 0.25, "$\\beta$", fontsize=7)
    c.set_xlabel("Time", fontsize=6.5, labelpad=1)
    legend(fig, "A", 1.06)
    save(fig, "plot_metrics_error.pdf")

    # ---------------- calibration
    fig, axes = plt.subplots(2, 1, figsize=(2.1, 2.45), gridspec_kw=dict(hspace=0.75))
    a, b = axes
    keys = "ABCD"
    base(a, "1-calibration at $t^*$", keys)
    a.axvline(ts, color=MUTED, lw=0.7, ls="--")
    for k in keys:
        tri(a, ts, S(k, ts), cols[k])
    a.text(ts + 0.12, 0.83, "group 1", fontsize=6)
    a.text(ts + 0.12, 0.1, "group 2", fontsize=6)
    base(b, "D-calibration", keys)
    b.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    for k in keys:
        tri(b, true[k], S(k, true[k]), cols[k])
    b.text(2.25, 0.72, "2 in [0.5, 1]", fontsize=6, ha="left")
    b.text(0.05, 0.22, "2 in [0, 0.5]", fontsize=6, ha="left")
    b.set_xlabel("Time", fontsize=6.5, labelpad=1)
    legend(fig, keys, 1.1)
    save(fig, "plot_metrics_calibration.pdf")


if __name__ == "__main__":
    mensa_results()
    mensa_ablation()
    baysurv()
    tmj()
    copula()
    margin_figures()
    survival_relations()
    als_counterfactuals()
    als_model_vs_km()
    metric_illustrations()
