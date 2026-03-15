import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

def visualize_unique_counts(df, column, color_palette="Blues_d", save_path=None, dpi=300):
    """
    Visualizes the unique counts of a specified column in a DataFrame,
    styled for research paper publication.

    Parameters:
    df         (pd.DataFrame): The input DataFrame.
    column     (str)         : The name of the column to visualize.
    color_palette (str)      : Seaborn palette name (default: 'Blues_d').
    save_path  (str|None)    : If provided, saves the figure to this path (e.g., 'fig.pdf').
    dpi        (int)         : Resolution for saved figure (default: 300).

    Returns:
    None: Displays (and optionally saves) a publication-quality bar plot.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    unique_counts = df[column].value_counts().sort_index()
    labels        = unique_counts.index.astype(str)
    values        = unique_counts.values
    x_pos         = np.arange(len(labels))

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family"      : "serif",          
        "font.serif"       : ["Times New Roman", "DejaVu Serif"],
        "font.size"        : 11,
        "axes.linewidth"   : 0.8,
        "axes.edgecolor"   : "#333333",
        "xtick.direction"  : "out",
        "ytick.direction"  : "out",
        "xtick.major.size" : 4,
        "ytick.major.size" : 4,
        "figure.dpi"       : 150,
    })

    palette = sns.color_palette(color_palette, len(labels))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))          # single-column ≈ 3.5 in; two-col ≈ 7 in

    bars = ax.bar(
        x_pos, values,
        color       = palette,
        width       = 0.6,
        edgecolor   = "white",
        linewidth   = 0.6,
        zorder      = 3,
    )

    # ── Value labels on bars ──────────────────────────────────────────────────
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.012,
            f"{val:,}",
            ha        = "center",
            va        = "bottom",
            fontsize  = 8.5,
            color     = "#333333",
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Count", fontsize=11, labelpad=8)
    ax.set_xlabel(column.replace("_", " ").title(), fontsize=11, labelpad=8)
    ax.set_title(
        f"Distribution of {column.replace('_', ' ').title()}",
        fontsize  = 13,
        fontweight= "bold",
        pad       = 12,
    )

    # Light horizontal grid only, pushed behind bars
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Remove top & right spines (clean look)
    sns.despine(ax=ax, top=True, right=True)

    # ── Save / Show ───────────────────────────────────────────────────────────
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to '{save_path}'")