r"""The look of the figures: colours and a matplotlib style, shared by compare.py and simulate.py."""

# Categorical colours, one per run, in a fixed order (colour-blind safe for the first three).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light (low) to dark (high)
INK, MUTED, FAINT = "#1a1a19", "#6b6a66", "#ebeae6"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#a8a6a0", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
}
