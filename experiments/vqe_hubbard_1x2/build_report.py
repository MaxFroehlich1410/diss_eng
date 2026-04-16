#!/usr/bin/env python3
"""Build plots and a LaTeX report for the VQE optimizer sweeps."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

_LOCAL_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_DIR / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qml_models.vqe import exact_ground_state_energy, build_hubbard_1x2_hamiltonian


HERE = _LOCAL_DIR
RESULTS_DIR = HERE / "results" / "optimizer_sweeps"
REPORT_DIR = HERE / "results" / "report"
FIGURES_DIR = REPORT_DIR / "figures"

OPTIMIZER_ORDER = ["adam", "bfgs", "qng", "krotov_hybrid"]
OPTIMIZER_LABELS = {
    "adam": "Adam",
    "bfgs": "BFGS",
    "qng": "QNG",
    "krotov_hybrid": "Hybrid Krotov",
}
OPTIMIZER_COLORS = {
    "adam": "#2563eb",
    "bfgs": "#0f766e",
    "qng": "#d97706",
    "krotov_hybrid": "#8b1e3f",
}
INSTANCE_ORDER = ["u2", "u4", "u8"]
INSTANCE_LABELS = {"u2": "$U=2$", "u4": "$U=4$", "u8": "$U=8$"}
INSTANCE_VALUES = {"u2": 2.0, "u4": 4.0, "u8": 8.0}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_analysis() -> dict[str, dict]:
    analyses = {}
    for optimizer in OPTIMIZER_ORDER:
        path = RESULTS_DIR / optimizer / f"analysis_{optimizer}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing analysis file: {path}")
        analyses[optimizer] = _load_json(path)
    return analyses


def _abs_err(value: float) -> float:
    return max(abs(float(value)), 1e-16)


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def _tex_escape(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def build_cross_optimizer_summary(analyses: dict[str, dict]) -> dict[str, dict[str, dict]]:
    summary: dict[str, dict[str, dict]] = {}
    for instance_key in INSTANCE_ORDER:
        instance_summary = {}
        for optimizer in OPTIMIZER_ORDER:
            instance_summary[optimizer] = analyses[optimizer]["instances"][instance_key]
        summary[instance_key] = instance_summary
    return summary


def plot_best_error_comparison(analyses: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(INSTANCE_ORDER))
    width = 0.18

    for idx, optimizer in enumerate(OPTIMIZER_ORDER):
        values = [
            _abs_err(analyses[optimizer]["instances"][instance]["best"]["error_mean"])
            for instance in INSTANCE_ORDER
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=OPTIMIZER_LABELS[optimizer],
            color=OPTIMIZER_COLORS[optimizer],
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"Best mean absolute energy error $|E - E_0|$")
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[key] for key in INSTANCE_ORDER])
    ax.set_title("Best energy accuracy by optimizer and interaction strength")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=2)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"best_error_comparison.{ext}")
    plt.close(fig)


def plot_best_wall_time_comparison(analyses: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(INSTANCE_ORDER))
    width = 0.18

    for idx, optimizer in enumerate(OPTIMIZER_ORDER):
        values = [
            analyses[optimizer]["instances"][instance]["best"]["wall_mean"]
            for instance in INSTANCE_ORDER
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=OPTIMIZER_LABELS[optimizer],
            color=OPTIMIZER_COLORS[optimizer],
        )

    ax.set_ylabel("Mean wall time of best configuration [s]")
    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[key] for key in INSTANCE_ORDER])
    ax.set_title("Best wall time by optimizer and interaction strength")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"best_wall_time_comparison.{ext}")
    plt.close(fig)


def plot_pareto(analyses: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)

    for ax, instance_key in zip(axes, INSTANCE_ORDER):
        for optimizer in OPTIMIZER_ORDER:
            best = analyses[optimizer]["instances"][instance_key]["best"]
            x_val = best["wall_mean"]
            y_val = _abs_err(best["error_mean"])
            ax.scatter(
                x_val,
                y_val,
                s=70,
                color=OPTIMIZER_COLORS[optimizer],
                label=OPTIMIZER_LABELS[optimizer],
            )
            ax.annotate(
                OPTIMIZER_LABELS[optimizer],
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Mean wall time [s]")
        ax.set_title(INSTANCE_LABELS[instance_key])
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Best mean absolute energy error $|E - E_0|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Accuracy vs wall-time trade-off for best hyperparameter settings", y=1.02)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"pareto_best_configs.{ext}")
    plt.close(fig)


def plot_adam_sensitivity(analyses: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = analyses["adam"]

    for instance_key in INSTANCE_ORDER:
        rows = data["instances"][instance_key]["all_configs"]
        rows = sorted(rows, key=lambda row: row["adam_lr"])
        ax.plot(
            [row["adam_lr"] for row in rows],
            [_abs_err(row["error_mean"]) for row in rows],
            marker="o",
            linewidth=2.0,
            label=INSTANCE_LABELS[instance_key],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Adam learning rate")
    ax.set_ylabel(r"Mean absolute final energy error $|E - E_0|$")
    ax.set_title("Adam sweep sensitivity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"adam_sensitivity.{ext}")
    plt.close(fig)


def plot_bfgs_sensitivity(analyses: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = analyses["bfgs"]

    for instance_key in INSTANCE_ORDER:
        rows = data["instances"][instance_key]["all_configs"]
        rows = sorted(rows, key=lambda row: row["bfgs_gtol"])
        ax.plot(
            [row["bfgs_gtol"] for row in rows],
            [_abs_err(row["error_mean"]) for row in rows],
            marker="o",
            linewidth=2.0,
            label=INSTANCE_LABELS[instance_key],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BFGS gradient tolerance")
    ax.set_ylabel(r"Mean absolute final energy error $|E - E_0|$")
    ax.set_title("BFGS sensitivity to convergence tolerance")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"bfgs_sensitivity.{ext}")
    plt.close(fig)


def plot_qng_heatmaps(analyses: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    data = analyses["qng"]
    lr_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    lam_values = [1e-4, 1e-3, 1e-2]
    vmax = max(
        math.log10(_abs_err(row["error_mean"]))
        for instance in INSTANCE_ORDER
        for row in data["instances"][instance]["all_configs"]
    )
    vmin = min(
        math.log10(_abs_err(row["error_mean"]))
        for instance in INSTANCE_ORDER
        for row in data["instances"][instance]["all_configs"]
    )

    im = None
    for ax, instance_key in zip(axes, INSTANCE_ORDER):
        rows = data["instances"][instance_key]["all_configs"]
        matrix = np.full((len(lr_values), len(lam_values)), np.nan)
        for row in rows:
            i = lr_values.index(row["qng_lr"])
            j = lam_values.index(row["qng_lam"])
            matrix[i, j] = math.log10(_abs_err(row["error_mean"]))

        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(lam_values)))
        ax.set_xticklabels([f"{value:g}" for value in lam_values])
        ax.set_yticks(range(len(lr_values)))
        ax.set_yticklabels([f"{value:g}" for value in lr_values])
        ax.set_xlabel(r"Regularization $\lambda$")
        ax.set_title(INSTANCE_LABELS[instance_key])
        if ax is axes[0]:
            ax.set_ylabel("QNG learning rate")

        for i in range(len(lr_values)):
            for j in range(len(lam_values)):
                error = 10 ** matrix[i, j]
                ax.text(j, i, f"{error:.1e}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\log_{10}|E - E_0|$")
    fig.suptitle("QNG sweep heatmaps", y=1.02)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"qng_heatmaps.{ext}")
    plt.close(fig)


def plot_krotov_heatmaps(analyses: dict[str, dict]) -> None:
    data = analyses["krotov_hybrid"]
    switch_values = [5, 10, 20]
    online_values = [0.03, 0.1, 0.3]
    batch_values = [0.1, 0.3, 1.0]

    fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.0), sharex=True, sharey=True)
    vmax = max(
        math.log10(max(_abs_err(row["error_mean"]), 1e-8))
        for instance in INSTANCE_ORDER
        for row in data["instances"][instance]["all_configs"]
    )
    vmin = min(
        math.log10(max(_abs_err(row["error_mean"]), 1e-8))
        for instance in INSTANCE_ORDER
        for row in data["instances"][instance]["all_configs"]
    )

    im = None
    for row_idx, instance_key in enumerate(INSTANCE_ORDER):
        rows = data["instances"][instance_key]["all_configs"]
        lookup = {
            (entry["switch"], entry["online_step"], entry["batch_step"]): entry
            for entry in rows
        }
        for col_idx, switch in enumerate(switch_values):
            ax = axes[row_idx, col_idx]
            matrix = np.full((len(online_values), len(batch_values)), np.nan)
            for i, online_step in enumerate(online_values):
                for j, batch_step in enumerate(batch_values):
                    entry = lookup[(switch, online_step, batch_step)]
                    matrix[i, j] = math.log10(max(_abs_err(entry["error_mean"]), 1e-8))

            im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(f"switch={switch}")
            if col_idx == 0:
                ax.set_ylabel(f"{INSTANCE_LABELS[instance_key]}\nonline step")
            if row_idx == len(INSTANCE_ORDER) - 1:
                ax.set_xlabel("batch step")

            ax.set_xticks(range(len(batch_values)))
            ax.set_xticklabels([f"{value:g}" for value in batch_values])
            ax.set_yticks(range(len(online_values)))
            ax.set_yticklabels([f"{value:g}" for value in online_values])

            for i, online_step in enumerate(online_values):
                for j, batch_step in enumerate(batch_values):
                    entry = lookup[(switch, online_step, batch_step)]
                    success = int(round(3 * entry["success_rate"]))
                    ax.text(
                        j,
                        i,
                        f"{entry['error_mean']:.1e}\n{success}/3",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_{10}|E - E_0|$")
    fig.suptitle("Hybrid Krotov sensitivity: error and success count per cell", y=1.01)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"krotov_heatmaps.{ext}")
    plt.close(fig)


def plot_all(analyses: dict[str, dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_best_error_comparison(analyses)
    plot_best_wall_time_comparison(analyses)
    plot_pareto(analyses)
    plot_adam_sensitivity(analyses)
    plot_bfgs_sensitivity(analyses)
    plot_qng_heatmaps(analyses)
    plot_krotov_heatmaps(analyses)


def _best_config_text(instance_summary: dict) -> str:
    names = instance_summary["param_names"]
    best = instance_summary["best"]
    parts = [f"{name}={best[name]}" for name in names]
    return ", ".join(parts)


def build_best_config_table(analyses: dict[str, dict]) -> str:
    lines = [
        r"\begin{tabular}{lllp{3.2cm}rrr}",
        r"\toprule",
        r"$U$ & Optimizer & Best config & Success & Mean $|E-E_0|$ & Mean wall [s] & Success rate \\",
        r"\midrule",
    ]
    for instance_key in INSTANCE_ORDER:
        first = True
        for optimizer in OPTIMIZER_ORDER:
            instance_summary = analyses[optimizer]["instances"][instance_key]
            best = instance_summary["best"]
            success = "yes" if best["success_rate"] >= 1.0 else "no"
            row = (
                f"{INSTANCE_VALUES[instance_key]:.0f} & "
                f"{OPTIMIZER_LABELS[optimizer]} & "
                f"{_tex_escape(_best_config_text(instance_summary))} & "
                f"{success} & "
                f"${_fmt_sci(_abs_err(best['error_mean']))}$ & "
                f"{best['wall_mean']:.3f} & "
                f"{best['success_rate']:.2f} \\\\"
            )
            if first:
                lines.append(row)
                first = False
            else:
                lines.append(row)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_ground_state_table() -> str:
    lines = [
        r"\begin{tabular}{rr}",
        r"\toprule",
        r"$U$ & Exact ground energy $E_0$ \\",
        r"\midrule",
    ]
    for instance_key in INSTANCE_ORDER:
        U = INSTANCE_VALUES[instance_key]
        e0 = exact_ground_state_energy(build_hubbard_1x2_hamiltonian(U))
        lines.append(f"{U:.0f} & {e0:.6f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_interpretation(analyses: dict[str, dict]) -> str:
    best_u4 = {
        optimizer: analyses[optimizer]["instances"]["u4"]["best"]
        for optimizer in OPTIMIZER_ORDER
    }
    best_u8 = {
        optimizer: analyses[optimizer]["instances"]["u8"]["best"]
        for optimizer in OPTIMIZER_ORDER
    }
    return rf"""
The sweep results separate the optimizer families cleanly. First, the benchmark
is not ansatz-limited for the tested interaction strengths: both BFGS and QNG
recover the exact half-filling ground energy to numerical precision for
$U \in \{{2,4,8\}}$. That means the 5-layer HV ansatz is expressive enough on
this 1$\times$2 instance, and the observed differences are genuinely
optimization differences rather than representational failure.

BFGS is the strongest overall method in this exact-statevector setting. It
reaches machine-precision energies for all three $U$ values with mean wall
times around $10^{{-2}}\,\mathrm{{s}}$, which is roughly one order of magnitude
faster than QNG and several times faster than Adam. The BFGS tolerance sweep is
also flat in practice: all tested values of \texttt{{bfgs\_gtol}} converge to the
same optimum, so the method is both fast and robust here.

QNG is almost as accurate as BFGS, but materially slower because every update
recomputes the full Fubini--Study metric tensor. The heatmaps show a broad
stable region at learning rates $0.03$ and $0.1$ across all three values of
$\lambda$. In contrast, larger learning rates such as $0.3$ and $1.0$ cause
clear breakdown, especially for $U=4$ and $U=8$. The main conclusion is that
QNG works extremely well once the learning rate is kept in the moderate regime,
but the metric overhead is not repaid on this tiny exact problem.

Adam behaves differently. It is consistently successful under the
$|E-E_0| \le 10^{{-3}}$ criterion, but it does not refine all the way to machine
precision within the fixed budget of 80 iterations. Its best mean errors remain
around $2\times 10^{{-4}}$ for all three interaction strengths. This makes Adam a
strong and fairly robust baseline, but not a precision winner on this task.
The preferred learning rate is also mildly interaction dependent: the sweep
selects \texttt{{adam\_lr}}$=0.1$ for $U=2$ and $U=4$, and
\texttt{{adam\_lr}}$=0.03$ for $U=8$.

Hybrid Krotov is the most model-dependent optimizer by far. At weak interaction
($U=2$) it has a narrow but very effective region and can also reach nearly
exact energies, with its best setting
(\texttt{{switch}}$=5$, \texttt{{online\_step}}$=0.03$,
\texttt{{batch\_step}}$=0.1$) yielding a mean error of only
${_fmt_sci(_abs_err(analyses['krotov_hybrid']['instances']['u2']['best']['error_mean']))}$.
However, this good behavior does not survive stronger interaction. For $U=4$ the
best hybrid-Krotov configuration still has mean error
${_fmt_sci(_abs_err(best_u4['krotov_hybrid']['error_mean']))}$ and zero successful
runs, while for $U=8$ the best error grows to
${_fmt_sci(_abs_err(best_u8['krotov_hybrid']['error_mean']))}$ with the entire
sweep failing the success criterion. The heatmaps show that the only potentially
viable batch step is the smallest tested value, \texttt{{batch\_step}}$=0.1$,
and that even then the online phase becomes destabilizing as $U$ grows.

The resulting ranking is therefore clear. In this exact dense benchmark, BFGS is
the best default optimizer because it is simultaneously the fastest and the most
accurate. QNG is an excellent second choice when one wants a geometry-aware
update rule and is willing to pay the metric-tensor overhead. Adam is a solid
first-order baseline that reaches chemically exact precision under the chosen
tolerance, but it underperforms BFGS and QNG on final refinement. Hybrid Krotov
is not competitive as a general-purpose optimizer here; its success at $U=2$
does show that the fixed-generator forward/backward interface is wired
correctly, but the sweep demonstrates that this optimizer is sharply
interaction- and hyperparameter-sensitive on the Hubbard VQE instance.
"""


def build_report_tex(analyses: dict[str, dict]) -> str:
    ground_table = build_ground_state_table()
    best_table = build_best_config_table(analyses)
    interpretation = build_interpretation(analyses)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{siunitx}}
\usepackage{{caption}}
\captionsetup{{font=small}}

\title{{Exact-Statevector VQE Optimizer Sweep Report}}
\author{{Codex-generated summary from local sweep outputs}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Benchmark Setup}}
The benchmark uses the exact-statevector 4-qubit, 5-layer Hamiltonian
Variational (HV) ansatz for the 1$\times$2 Fermi--Hubbard model at half
filling. The qubit ordering is
$(q_0,q_1,q_2,q_3)=({{\rm site}}\ 1,\uparrow;\ {{\rm site}}\ 2,\uparrow;\ {{\rm site}}\ 1,\downarrow;\ {{\rm site}}\ 2,\downarrow)$.
The physical Hamiltonian is
\[
H(U,t)=t\,H_{{\rm hop,unit}}+U\,H_{{\rm onsite,unit}}, \qquad t=-1,
\]
with
\[
H_{{\rm hop,unit}}=\tfrac12(X_0X_1+Y_0Y_1)+\tfrac12(X_2X_3+Y_2Y_3)
\]
and
\[
H_{{\rm onsite,unit}}=\lvert 11\rangle\langle 11\rvert_{{(0,2)}}+\lvert 11\rangle\langle 11\rvert_{{(1,3)}}.
\]
The reference state is the exact half-filling ground state of the noninteracting
($U=0$) Hamiltonian, obtained by exact diagonalization in the fixed
two-particle sector. The variational circuit applies five repeated layers
\[
U_{{\rm layer}}(\phi_\ell,\tau_\ell)=
\exp(+i\tau_\ell H_{{\rm hop,unit}})
\exp(+i\phi_\ell H_{{\rm onsite,unit}}),
\]
with parameter vector
\[
\theta=(\phi_1,\tau_1,\phi_2,\tau_2,\phi_3,\tau_3,\phi_4,\tau_4,\phi_5,\tau_5).
\]
All losses are exact expectation values
\[
E(\theta)=\langle \psi(\theta)\rvert H \lvert \psi(\theta)\rangle
\]
evaluated with dense linear algebra only. No shots, no sampling noise, no
hardware noise, and no finite-difference black-box gradients are used.

The sweep covered $U\in\{{2,4,8\}}$, three random seeds per configuration, and
80 outer iterations for Adam, QNG, and hybrid Krotov. The success criterion was
$|E-E_0|\le 10^{{-3}}$, where $E_0$ is the exact half-filling ground energy.

\begin{{table}}[H]
\centering
{ground_table}
\caption{{Exact half-filling ground energies used as references.}}
\end{{table}}

\section*{{Optimizer Grids}}
The tested hyperparameter grids were:
\begin{{itemize}}
\item Adam: \texttt{{adam\_lr}} $\in \{{0.01,0.03,0.1,0.3,1.0\}}$.
\item BFGS: \texttt{{bfgs\_gtol}} $\in \{{10^{{-5}},10^{{-7}},10^{{-9}}\}}$.
\item QNG: \texttt{{qng\_lr}} $\in \{{0.01,0.03,0.1,0.3,1.0\}}$ and
      \texttt{{qng\_lam}} $\in \{{10^{{-4}},10^{{-3}},10^{{-2}}\}}$.
\item Hybrid Krotov: \texttt{{switch}} $\in \{{5,10,20\}}$,
      \texttt{{online\_step}} $\in \{{0.03,0.1,0.3\}}$, and
      \texttt{{batch\_step}} $\in \{{0.1,0.3,1.0\}}$.
\end{{itemize}}

\section*{{Best Configurations}}
\begin{{table}}[H]
\centering
\small
{best_table}
\caption{{Best configuration found for each optimizer and interaction strength.
The error column reports the mean absolute energy error over the three seeds.}}
\end{{table}}

\section*{{Cross-Optimizer Comparison}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/best_error_comparison.pdf}}
\caption{{Best mean absolute energy error achieved by each optimizer family at each interaction strength.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/best_wall_time_comparison.pdf}}
\caption{{Mean wall time of the best hyperparameter configuration in each optimizer family.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/pareto_best_configs.pdf}}
\caption{{Accuracy--runtime trade-off of the best configuration from each optimizer family, shown separately for each interaction strength.}}
\end{{figure}}

\section*{{Hyperparameter Sensitivity}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/adam_sensitivity.pdf}}
\caption{{Adam sensitivity to learning rate.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/bfgs_sensitivity.pdf}}
\caption{{BFGS sensitivity to the stopping tolerance. The flat curves indicate that the method is robust on this benchmark.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/qng_heatmaps.pdf}}
\caption{{QNG sweep heatmaps. The broad low-error region at learning rates $0.03$ and $0.1$ explains the method's robustness once the step size is kept moderate.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/krotov_heatmaps.pdf}}
\caption{{Hybrid Krotov sweep heatmaps. Each cell shows mean error and the number of successful seeds out of three.}}
\end{{figure}}

\section*{{Interpretation}}
{interpretation}

\section*{{Takeaway}}
For this exact 4-qubit Hubbard VQE benchmark, BFGS is the strongest default
choice: it is both the fastest and the most accurate across all tested
interaction strengths. QNG also reaches the exact optimum but is slower because
of the metric-tensor overhead. Adam is a strong practical baseline when one only
needs $\sim 10^{{-4}}$ accuracy under a fixed iteration budget. Hybrid Krotov is
not robust on the strongly interacting instances and should therefore not be
treated as a competitive all-purpose optimizer for this benchmark without a
substantially different schedule or update rule.

\end{{document}}
"""


def write_report(tex_source: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = REPORT_DIR / "vqe_optimizer_sweep_report.tex"
    tex_path.write_text(tex_source)
    return tex_path


def maybe_compile_pdf(tex_path: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("pdflatex not found; wrote LaTeX source only.")
        return

    for _ in range(2):
        subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={REPORT_DIR}",
                str(tex_path),
            ],
            cwd=REPORT_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(f"Compiled PDF: {REPORT_DIR / 'vqe_optimizer_sweep_report.pdf'}")


def main() -> None:
    analyses = load_analysis()
    plot_all(analyses)
    tex_source = build_report_tex(analyses)
    tex_path = write_report(tex_source)
    print(f"Wrote LaTeX report to {tex_path}")
    maybe_compile_pdf(tex_path)


if __name__ == "__main__":
    main()
