"""Figures and summary tables from results/metrics. Only plots data that exists; missing figures are skipped.

Usage (from the repository root):
    python -m analysis.visualize                     # primary protocol (strict), stage 'post' (after adaptation)
    python -m analysis.visualize --stage pre         # same figures for representations before adaptation
    python -m analysis.visualize --include-smoke     # also plot runs tagged *_smoke (pipeline tests only)

Figures (results/figures/*.png and *.pdf):
  fig1_additive_verification   ORBIT verification with the additive baseline (ours vs numbers printed in the paper)
  fig2_pretrained_vs_random    150M pretrained vs random: accessibility vs relative depth, one panel per order
  fig3_capacity                pretrained 8M / 35M / 150M: accessibility vs relative depth, one panel per order
  fig4_layerwise_by_order      one panel per model, one line per interaction order, absolute layer index
  fig5_heatmaps                layer x order accessibility heatmaps per model
  fig6_prediction_vs_access    test Spearman (fitness prediction) vs final-layer accessibility, one point per run
  fig7_before_after            accessibility before (dashed) vs after (solid) fitness adaptation
  fig8_protocol_sensitivity    strict vs random-cube probe split (leakage check), if both exist
Lines are the mean over seeds with a +-1 s.d. band when more than one seed exists. Heatmap colors are clipped to [-1, 1].
Summary table: results/metrics/summary_table.csv
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from methods.gb1 import FIGURES_DIR, METRICS_DIR

ORDER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#6250d6"}          # validated categorical slots
ORDER_NAMES = {1: "order 1 (single)", 2: "order 2 (pairwise)", 3: "order 3", 4: "order 4"}
MODEL_STYLE = {                                   # label -> (color, marker); same in every figure
    "8M pretrained": ("#6250d6", "o"), "35M pretrained": ("#1baf7a", "s"),
    "150M pretrained": ("#2a78d6", "^"), "150M random": ("#eb6834", "D")}
DIVERGING = LinearSegmentedColormap.from_list("orange_gray_blue", ["#eb6834", "#ebebe8", "#2a78d6"])


# ---------------- loading ----------------
def load_accessibility(include_smoke):
    frames = []
    for path in sorted(METRICS_DIR.glob("accessibility_*.csv")):
        if "_smoke" in path.stem and not include_smoke:
            continue
        frame = pd.read_csv(path)
        frame["condition"] = path.stem.replace("accessibility_", "")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["model"] = data["size"] + " " + data["init"]
    return data


def mean_sd(frame, group_columns):
    """Mean and s.d. (NaN -> 0 for a single seed) of accessibility across seeds."""
    grouped = frame.groupby(group_columns)["accessibility_r2"]
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result["std"] = result["std"].fillna(0.0)
    return result


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def style_axis(axis, ylabel=None, xlabel=None, title=None):
    axis.axhline(0, color="#888", lw=0.8, zorder=0)                      # R^2 = 0: no better than the mean
    axis.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if ylabel:
        axis.set_ylabel(ylabel)
    if xlabel:
        axis.set_xlabel(xlabel)
    if title:
        axis.set_title(title, fontsize=10)


def plot_line(axis, frame, x_column, color, label, marker="o", linestyle="-"):
    curve = mean_sd(frame, [x_column]).sort_values(x_column)
    axis.plot(curve[x_column], curve["mean"], color=color, marker=marker, ms=4, lw=1.8, ls=linestyle, label=label)
    if curve["count"].max() > 1:
        axis.fill_between(curve[x_column], curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=color, alpha=0.15, lw=0)


def order_panels(title, models, data, x_column="relative_depth", linestyles=None):
    """Four panels (one per order) with one line per model."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for order, axis in zip(range(1, 5), axes):
        for model in models:
            subset = data[(data["model"] == model) & (data["order"] == order)]
            if len(subset):
                color, marker = MODEL_STYLE[model]
                plot_line(axis, subset, x_column, color, model, marker)
        style_axis(axis, xlabel="relative depth (layer / last layer)" if x_column == "relative_depth" else "layer",
                   title=ORDER_NAMES[order], ylabel="accessibility (held-out R$^2$)" if order == 1 else None)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, y=1.03)
    return fig


# ---------------- figures ----------------
def fig1_additive():
    path = METRICS_DIR / "additive_baseline.json"
    if not path.exists():
        return print("skipped fig1: run python -m methods.additive_baseline")
    report = json.loads(path.read_text())
    ours, paper = report["our_reproduction"], report["reported_in_orbit_paper"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    orders = np.arange(1, 5)
    left.bar(orders - 0.2, [ours["functional_recovery_r2_probe_test_cubes"][str(k)] for k in orders], 0.4, color="#2a78d6", label="our reproduction")
    left.bar(orders + 0.2, [paper["functional_recovery_r2"][str(k)] for k in orders], 0.4, color="#9aa0a6", label="reported in ORBIT paper")
    left.set_xticks(orders, [f"order {k}" for k in orders])
    style_axis(left, ylabel="functional recovery $R^2_k$ (613 probe-test cubes)", title="Additive ridge model: recovered Walsh coefficients")
    left.legend(frameon=False, fontsize=8)
    names = ["test_spearman", "test_r2"]
    positions = np.arange(len(names))
    right.bar(positions - 0.2, [ours[n] for n in names], 0.4, color="#2a78d6")
    right.bar(positions + 0.2, [paper[n] for n in names], 0.4, color="#9aa0a6")
    right.set_xticks(positions, ["Spearman", "R$^2$"])
    style_axis(right, title="FLIP test (3-4 mutants)")
    fig.text(0.5, -0.04, "max |order 2-4 coefficient| of additive predictions over all cubes = %.1e (must be ~0)"
             % ours["max_abs_predicted_order_2_to_4_coefficient"], ha="center", fontsize=8)
    save(fig, "fig1_additive_verification")


def fig2_pretrained_vs_random(data):
    models = ["150M pretrained", "150M random"]
    if data["model"].isin(models).sum() == 0 or not set(models) <= set(data["model"]):
        return print("skipped fig2: need both 150M pretrained and 150M random results")
    save(order_panels("Pretrained vs randomly initialized ESM-2 150M", models, data), "fig2_pretrained_vs_random")


def fig3_capacity(data):
    models = [m for m in ["8M pretrained", "35M pretrained", "150M pretrained"] if m in set(data["model"])]
    if len(models) < 2:
        return print("skipped fig3: need at least two pretrained sizes")
    save(order_panels("Capacity: pretrained ESM-2 8M / 35M / 150M", models, data), "fig3_capacity")


def fig4_layerwise(data):
    models = [m for m in MODEL_STYLE if m in set(data["model"])]
    if not models:
        return print("skipped fig4: no data")
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for order in range(1, 5):
            subset = data[(data["model"] == model) & (data["order"] == order)]
            plot_line(axis, subset, "layer", ORDER_COLORS[order], ORDER_NAMES[order])
        style_axis(axis, xlabel="layer (0 = embeddings)", title=model, ylabel="accessibility (held-out R$^2$)" if axis is axes[0, 0] else None)
    axes[0, 0].legend(fontsize=8, frameon=False)
    save(fig, "fig4_layerwise_by_order")


def fig5_heatmaps(data):
    models = [m for m in MODEL_STYLE if m in set(data["model"])]
    if not models:
        return print("skipped fig5: no data")
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 2.1 * len(models) + 0.6), squeeze=False)
    for axis, model in zip(axes[:, 0], models):
        table = data[data["model"] == model].groupby(["order", "layer"])["accessibility_r2"].mean().unstack("layer")
        image = axis.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        axis.set_yticks(range(len(table)), [f"order {k}" for k in table.index])
        axis.set_xticks(range(table.shape[1]), table.columns)
        axis.set_title(model, fontsize=10, loc="left")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if table.shape[1] <= 16:
                    axis.text(j, i, f"{table.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axes[-1, 0].set_xlabel("layer (0 = embeddings)")
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="accessibility R$^2$ (clipped to [-1, 1])", shrink=0.8)
    save(fig, "fig5_heatmaps")


def load_training_metrics():
    rows = []
    for path in METRICS_DIR.glob("train_*.json"):
        result = json.loads(path.read_text())
        rows.append({"run": result["run"], "size": result["size"], "init": result["init"], "seed": result["seed"],
                     "test_spearman": result["test"]["spearman"], "test_r2": result["test"]["r2"],
                     "total_params": result["total_params"], "trainable_params": result["trainable_params"],
                     "training_time_seconds": result["training_time_seconds"]})
    return pd.DataFrame(rows)


def fig6_prediction_vs_accessibility(data, training):
    if training.empty or data.empty:
        return print("skipped fig6: need training metrics and accessibility results")
    final = data[data["layer"] == data.groupby("condition")["layer"].transform("max")]
    final = final.merge(training[["size", "init", "seed", "test_spearman"]], on=["size", "init", "seed"])
    if final.empty:
        return print("skipped fig6: no accessibility result matches a trained run")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for order, axis in zip(range(1, 5), axes):
        for model, (color, marker) in MODEL_STYLE.items():
            subset = final[(final["model"] == model) & (final["order"] == order)]
            axis.scatter(subset["test_spearman"], subset["accessibility_r2"], color=color, marker=marker, s=40, label=model, edgecolor="white", lw=0.8)
        style_axis(axis, xlabel="FLIP test Spearman", title=ORDER_NAMES[order], ylabel="final-layer accessibility R$^2$" if order == 1 else None)
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, "fig6_prediction_vs_accessibility")


def fig7_before_after(all_data):
    models = [m for m in MODEL_STYLE if {"pre", "post"} <= set(all_data[all_data["model"] == m]["stage"])]
    if not models:
        return print("skipped fig7: need both stage 'pre' and 'post' for a model")
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for order in range(1, 5):
            for stage, linestyle in [("pre", "--"), ("post", "-")]:
                subset = all_data[(all_data["model"] == model) & (all_data["order"] == order) & (all_data["stage"] == stage)]
                plot_line(axis, subset, "layer", ORDER_COLORS[order], f"{ORDER_NAMES[order]}, {'before' if stage == 'pre' else 'after'}", linestyle=linestyle)
        style_axis(axis, xlabel="layer (0 = embeddings)", title=model, ylabel="accessibility (held-out R$^2$)" if axis is axes[0, 0] else None)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    save(fig, "fig7_before_after")


def fig8_protocol_sensitivity(include_smoke):
    strict, random = None, None
    frames = []
    for path in sorted(METRICS_DIR.glob("accessibility_*_random.csv")):
        if "_smoke" in path.stem and not include_smoke:
            continue
        frames.append(pd.read_csv(path).assign(condition=path.stem))
    if not frames:
        return print("skipped fig8: no accessibility_*_random.csv (optional sensitivity analysis)")
    random = pd.concat(frames)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for order, axis in zip(range(1, 5), axes):
        for protocol, frame, linestyle in [("strict (primary)", None, "-"), ("random cubes (leaky)", random, "--")]:
            if frame is None:
                frame = pd.concat([pd.read_csv(p).assign(condition=p.stem) for p in sorted(METRICS_DIR.glob("accessibility_*_strict.csv"))
                                   if "_smoke" not in p.stem or include_smoke])
            frame = frame[frame["order"] == order]
            plot_line(axis, frame, "relative_depth", "#2a78d6" if linestyle == "-" else "#eb6834", protocol, linestyle=linestyle)
        style_axis(axis, xlabel="relative depth", title=ORDER_NAMES[order], ylabel="accessibility R$^2$" if order == 1 else None)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Probe-split sensitivity (all conditions averaged): random splits share variants between probe train and test", y=1.03)
    save(fig, "fig8_protocol_sensitivity")


def summary_table(data, training):
    """One row per condition and order: final-layer and best-layer accessibility (mean over seeds)."""
    rows = []
    for (model, stage, protocol, order), group in data.groupby(["model", "stage", "protocol", "order"]):
        by_layer = group.groupby("layer")["accessibility_r2"].mean()
        final_layer = by_layer.index.max()
        rows.append({"model": model, "stage": stage, "protocol": protocol, "order": order,
                     "n_seeds": group["seed"].nunique(), "final_layer_accessibility": by_layer[final_layer],
                     "best_layer": by_layer.idxmax(), "best_layer_accessibility": by_layer.max()})
    table = pd.DataFrame(rows)
    if not training.empty:
        training = training.assign(model=training["size"] + " " + training["init"])
        fitness = training.groupby("model")[["test_spearman", "test_r2", "trainable_params", "total_params"]].mean().reset_index()
        table = table.merge(fitness, on="model", how="left")
    table.round(4).to_csv(METRICS_DIR / "summary_table.csv", index=False)
    print(table.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["pre", "post"], default="post")
    parser.add_argument("--protocol", choices=["strict", "random"], default="strict")
    parser.add_argument("--pooling", choices=["sites", "mean"], default="sites")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})

    fig1_additive()
    everything = load_accessibility(args.include_smoke)
    if everything.empty:
        return print("no accessibility results yet: run models/ and analysis/accessibility.py first")
    everything = everything[(everything["protocol"] == args.protocol) & (everything["pooling"] == args.pooling)]
    data = everything[everything["stage"] == args.stage]
    training = load_training_metrics()
    if not args.include_smoke:
        training = training[~training["run"].str.contains("_smoke")]
    fig2_pretrained_vs_random(data)
    fig3_capacity(data)
    fig4_layerwise(data)
    fig5_heatmaps(data)
    fig6_prediction_vs_accessibility(data, training)
    fig7_before_after(everything)
    fig8_protocol_sensitivity(args.include_smoke)
    summary_table(everything, training)


if __name__ == "__main__":
    main()
