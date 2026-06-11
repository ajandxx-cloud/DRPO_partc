#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ROOT / "Experiments" / "analysis" / "drpo_sensitivity_oat_3_11_full" / "stage1_summary_enhanced.csv"
LEGACY_INPUT_CSV = ROOT / "Experiments" / "analysis" / "dspo_plus_spo_sensitivity_oat_3_11_full" / "stage1_summary_enhanced.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "images"

TAB10 = plt.get_cmap("tab10").colors
BLUE = TAB10[0]
ORANGE = TAB10[1]
GREEN = TAB10[2]
RED = TAB10[3]
PURPLE = TAB10[4]
GRAY = TAB10[7]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


PLOT_CONFIGS = [
    {
        "factor": "incentive_sens",
        "default_raw": -0.25,
        "transform_x": lambda v: abs(v),
        "sort_key": lambda v: abs(v),
        "xlabel": r"$\beta^{price}$",
        "title": r"Effect of price sensitivity $\beta^{price}$",
        "stem": "fig11_beta_price_sensitivity",
    },
    {
        "factor": "outside_option_util",
        "default_raw": -1.0,
        "transform_x": lambda v: v,
        "sort_key": lambda v: v,
        "xlabel": r"$V_0$",
        "title": r"Effect of outside-option utility $V_0$",
        "stem": "fig12_outside_option_sensitivity",
    },
    {
        "factor": "home_util",
        "default_raw": 1.4,
        "transform_x": lambda v: v,
        "sort_key": lambda v: v,
        "xlabel": r"$\mathrm{ASC}_{home}$",
        "title": r"Effect of home-pickup preference $\mathrm{ASC}_{home}$",
        "stem": "fig13_home_preference_sensitivity",
    },
]


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key):
    return float(row[key])


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#444444", length=3, width=0.8)
    ax.grid(axis="y", linestyle=(0, (1.5, 2.2)), linewidth=0.7, color="#D9D9D9")
    ax.set_facecolor("white")


def plot_factor(rows, cfg):
    factor_rows = [r for r in rows if r["factor"] == cfg["factor"]]
    factor_rows.sort(key=lambda r: cfg["sort_key"](float(r["value"])))

    default_row = next(r for r in factor_rows if float(r["value"]) == cfg["default_raw"])
    default_profit = to_float(default_row, "net_profit_mean")
    default_cost = to_float(default_row, "total_costs_mean")

    x = [cfg["transform_x"](float(r["value"])) for r in factor_rows]
    profit_change = [100.0 * (to_float(r, "net_profit_mean") - default_profit) / default_profit for r in factor_rows]
    cost_change = [100.0 * (to_float(r, "total_costs_mean") - default_cost) / default_cost for r in factor_rows]
    profit_low = [100.0 * (to_float(r, "net_profit_ci95_low") - default_profit) / default_profit for r in factor_rows]
    profit_high = [100.0 * (to_float(r, "net_profit_ci95_high") - default_profit) / default_profit for r in factor_rows]
    cost_low = [100.0 * (to_float(r, "total_costs_ci95_low") - default_cost) / default_cost for r in factor_rows]
    cost_high = [100.0 * (to_float(r, "total_costs_ci95_high") - default_cost) / default_cost for r in factor_rows]
    quit_rate = [to_float(r, "quit_rate_mean") for r in factor_rows]
    quit_low = [to_float(r, "quit_rate_ci95_low") for r in factor_rows]
    quit_high = [to_float(r, "quit_rate_ci95_high") for r in factor_rows]
    default_x = cfg["transform_x"](cfg["default_raw"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    ax_top, ax_bottom = axes

    ax_top.fill_between(x, profit_low, profit_high, color=BLUE, alpha=0.12, linewidth=0)
    ax_top.fill_between(x, cost_low, cost_high, color=ORANGE, alpha=0.12, linewidth=0)
    ax_top.plot(
        x,
        profit_change,
        color=BLUE,
        marker="o",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=2.0,
        label="Net profit",
    )
    ax_top.plot(
        x,
        cost_change,
        color=ORANGE,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linestyle="--",
        linewidth=2.0,
        label="Total costs",
    )
    ax_top.axvline(default_x, color=GRAY, linestyle=(0, (2, 2)), linewidth=1.1)
    ax_top.axhline(0.0, color="#CFCFCF", linewidth=0.9)
    ax_top.set_ylabel("Change vs. default (%)")
    ax_top.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.0)
    ax_top.text(0.01, 0.97, "(a)", transform=ax_top.transAxes, va="top", ha="left", fontweight="bold")

    ax_bottom.fill_between(x, quit_low, quit_high, color=GREEN, alpha=0.12, linewidth=0)
    ax_bottom.plot(
        x,
        quit_rate,
        color=GREEN,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=2.0,
    )
    ax_bottom.axvline(default_x, color=GRAY, linestyle=(0, (2, 2)), linewidth=1.1)
    ax_bottom.set_ylabel("Quit rate (%)")
    ax_bottom.set_xlabel(cfg["xlabel"])
    ax_bottom.text(0.01, 0.95, "(b)", transform=ax_bottom.transAxes, va="top", ha="left", fontweight="bold")

    for ax in axes:
        style_axis(ax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.align_ylabels(axes)

    png_path = cfg["output_dir"] / f"{cfg['stem']}.png"
    pdf_path = cfg["output_dir"] / f"{cfg['stem']}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    p = argparse.ArgumentParser(description="Plot the main sensitivity figures for the paper.")
    p.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser()
    if not input_csv.is_absolute():
        input_csv = (Path.cwd() / input_csv).resolve()
    if not input_csv.exists() and LEGACY_INPUT_CSV.exists():
        input_csv = LEGACY_INPUT_CSV
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_csv)
    outputs = []
    for cfg in PLOT_CONFIGS:
        cfg_local = dict(cfg)
        cfg_local["output_dir"] = output_dir
        outputs.extend(plot_factor(rows, cfg_local))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
