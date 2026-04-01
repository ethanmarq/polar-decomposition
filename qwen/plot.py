"""
Reproduces Figure 5 and Figure C.12 from PolarGrad Section 6.4.

Usage:
    python plot_qwen.py \
        --adamw        results/qwen_adamw_lr0.001.json \
        --muon_adamw   results/qwen_muon_adamw_lr0.001.json \
        --muon_polar   results/qwen_muon_polarsgdm_lr0.001.json \
        --outdir       figures/

Outputs:
    figure5.pdf       — training loss + κ₂ embed + κ₂ head  (3 panels)
    figure_c12.pdf    — nuclear norm embed + nuclear norm head (2 panels)
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paper style ───────────────────────────────────────────────────────────────
LABELS = {
    "adamw":          "AdamW",
    "muon_adamw":     "Muon + AdamW",
    "muon_polarsgdm": "Muon + PolarSGDM",
}
COLORS = {
    "adamw":          "#1f77b4",   # blue
    "muon_adamw":     "#ff7f0e",   # orange
    "muon_polarsgdm": "#2ca02c",   # green
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  11,
    "axes.titlesize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "lines.linewidth": 1.2,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def savefig(fig, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    bbox_inches="tight")
    print(f"  saved {name}.pdf / .png")

# ── Figure 5 ──────────────────────────────────────────────────────────────────
def plot_figure5(runs: dict, outdir: str):
    """
    3-panel figure matching paper Figure 5:
      left   – training loss (linear y)
      centre – κ₂(∇f(W^embed, ξ))  (log y)
      right  – κ₂(∇f(W^head,  ξ))  (log y)
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    for key, d in runs.items():
        steps = d["steps"]
        kw = dict(label=LABELS[key], color=COLORS[key])
        axes[0].plot(steps, d["loss"],       **kw)
        axes[1].semilogy(steps, d["cond_embed"], **kw)
        axes[2].semilogy(steps, d["cond_head"],  **kw)

    # ── axes 0: training loss ──────────────────────────────────────────────
    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel("training loss")
    axes[0].set_xlim(left=0)

    # ── axes 1: κ₂ embed ──────────────────────────────────────────────────
    axes[1].set_xlabel("iteration $k$")
    axes[1].set_ylabel(
        r"$\kappa_2\!\left(\nabla f(W^{\mathrm{embed}}_k,\,\xi_k)\right)$"
    )
    axes[1].set_xlim(left=0)
    axes[1].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))

    # ── axes 2: κ₂ head ───────────────────────────────────────────────────
    axes[2].set_xlabel("iteration $k$")
    axes[2].set_ylabel(
        r"$\kappa_2\!\left(\nabla f(W^{\mathrm{head}}_k,\,\xi_k)\right)$"
    )
    axes[2].set_xlim(left=0)
    axes[2].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))

    # shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.suptitle("Figure 5: Qwen2.5 Pre-Training (Section 6.4)", y=1.01)
    fig.tight_layout()
    savefig(fig, outdir, "figure5")
    plt.close(fig)

# ── Figure C.12 ───────────────────────────────────────────────────────────────
def plot_figure_c12(runs: dict, outdir: str):
    """
    2-panel figure matching paper Figure C.12:
      left  – nuclear norm of ∇f(W^embed, ξ)  (log y)
      right – nuclear norm of ∇f(W^head,  ξ)  (log y)
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    for key, d in runs.items():
        steps = d["steps"]
        kw = dict(label=LABELS[key], color=COLORS[key])
        axes[0].semilogy(steps, d["nuc_embed"], **kw)
        axes[1].semilogy(steps, d["nuc_head"],  **kw)

    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel(
        r"$\|\nabla f(W^{\mathrm{embed}}_k,\,\xi_k)\|_{\mathrm{nuc}}$"
    )
    axes[0].set_xlim(left=0)

    axes[1].set_xlabel("iteration $k$")
    axes[1].set_ylabel(
        r"$\|\nabla f(W^{\mathrm{head}}_k,\,\xi_k)\|_{\mathrm{nuc}}$"
    )
    axes[1].set_xlim(left=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    fig.suptitle("Figure C.12: Gradient Nuclear Norms (Section C.4)", y=1.01)
    fig.tight_layout()
    savefig(fig, outdir, "figure_c12")
    plt.close(fig)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce PolarGrad Section 6.4 figures from JSON results."
    )
    parser.add_argument("--adamw",      required=True,
                        help="JSON from --optimizer adamw run")
    parser.add_argument("--muon_adamw", required=True,
                        help="JSON from --optimizer muon_adamw run")
    parser.add_argument("--muon_polar", required=True,
                        help="JSON from --optimizer muon_polarsgdm run")
    parser.add_argument("--outdir",     default="figures",
                        help="Directory to write figures (default: figures/)")
    args = parser.parse_args()

    runs = {
        "adamw":          load(args.adamw),
        "muon_adamw":     load(args.muon_adamw),
        "muon_polarsgdm": load(args.muon_polar),
    }

    # Validate all required keys are present
    required = {"steps", "loss", "cond_embed", "cond_head", "nuc_embed", "nuc_head"}
    for name, d in runs.items():
        missing = required - d.keys()
        if missing:
            raise ValueError(f"{name} JSON is missing keys: {missing}")

    print("Generating figures...")
    plot_figure5(runs, args.outdir)
    plot_figure_c12(runs, args.outdir)
    print(f"Done. Figures written to {args.outdir}/")

if __name__ == "__main__":
    main()
