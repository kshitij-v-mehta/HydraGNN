#!/usr/bin/env python
# coding: utf-8
"""
Periodic table heatmap of element frequencies across HydraGNN datasets.

Reads `trainset/x[:,0]` (atomic numbers) from each ADIOS2 .bp file,
aggregates counts across all datasets, and saves two PDFs:
  - periodic_table_heatmap_<N>sets.pdf       (linear scale)
  - periodic_table_heatmap_<N>sets_log.pdf   (log scale)

To add datasets, append names to DATASET_LIST below.
"""

import os
import numpy as np
from adios2 import Stream
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# IEEE publication settings:
#   - Full-width (two-column spanning) figure: 7.16 in wide
#   - Body text: 9pt  →  axis labels / colorbar: 8pt, element symbols: 5pt,
#     atomic numbers: 4pt, title: 9pt
plt.rcParams.update({
    "font.size":        8,
    "font.family":      "serif",        # IEEE uses Times / serif fonts
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,             # embed fonts as Type-42 (TrueType) in PDF
    "ps.fonttype":      42,
})

# ---------------------------------------------------------------------------
# Config  —  edit these as needed
# ---------------------------------------------------------------------------

DIRNAME = "/lustre/orion/lrn070/world-shared/kmehta/hydragnn/datasets/v2"

DATASET_LIST = [
    "Alexandria-v2",
    "ANI1x-v2",
    "MPTrj-v2",
    "Nabla2DFT-v2",
    "OC2020-v2",
    "OC2022-v2",
    "OC25-v2",
    "ODAC23-v2",
    "OMat24-v2",
    "OMol25-v2",
    "OPoly2026-v2",
    "QCML-v2",
    "QM7X-v2",
    "transition1x-v2",
]

# ---------------------------------------------------------------------------
# Periodic table metadata
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
]

# (row, col) positions — 0-indexed, 18-column layout.
# Lanthanides (58-71) and actinides (90-103) sit in the detached f-block rows 7 and 8.
PERIODIC_TABLE_LAYOUT = {
    1:  (0,  0),  2:  (0, 17),
    3:  (1,  0),  4:  (1,  1),  5:  (1, 12),  6:  (1, 13),  7:  (1, 14),
    8:  (1, 15),  9:  (1, 16), 10:  (1, 17),
    11: (2,  0), 12:  (2,  1), 13:  (2, 12), 14:  (2, 13), 15:  (2, 14),
    16: (2, 15), 17:  (2, 16), 18:  (2, 17),
    19: (3,  0), 20:  (3,  1), 21:  (3,  2), 22:  (3,  3), 23:  (3,  4),
    24: (3,  5), 25:  (3,  6), 26:  (3,  7), 27:  (3,  8), 28:  (3,  9),
    29: (3, 10), 30:  (3, 11), 31:  (3, 12), 32:  (3, 13), 33:  (3, 14),
    34: (3, 15), 35:  (3, 16), 36:  (3, 17),
    37: (4,  0), 38:  (4,  1), 39:  (4,  2), 40:  (4,  3), 41:  (4,  4),
    42: (4,  5), 43:  (4,  6), 44:  (4,  7), 45:  (4,  8), 46:  (4,  9),
    47: (4, 10), 48:  (4, 11), 49:  (4, 12), 50:  (4, 13), 51:  (4, 14),
    52: (4, 15), 53:  (4, 16), 54:  (4, 17),
    55: (5,  0), 56:  (5,  1), 57:  (5,  2), 72:  (5,  3), 73:  (5,  4),
    74: (5,  5), 75:  (5,  6), 76:  (5,  7), 77:  (5,  8), 78:  (5,  9),
    79: (5, 10), 80:  (5, 11), 81:  (5, 12), 82:  (5, 13), 83:  (5, 14),
    84: (5, 15), 85:  (5, 16), 86:  (5, 17),
    87: (6,  0), 88:  (6,  1), 89:  (6,  2),104:  (6,  3),105:  (6,  4),
    106:(6,  5),107:  (6,  6),108:  (6,  7),109:  (6,  8),110:  (6,  9),
    111:(6, 10),112:  (6, 11),113:  (6, 12),114:  (6, 13),115:  (6, 14),
    116:(6, 15),117:  (6, 16),118:  (6, 17),
    # Lanthanides — f-block row 7
     58: (7,  2),  59: (7,  3),  60: (7,  4),  61: (7,  5),  62: (7,  6),
     63: (7,  7),  64: (7,  8),  65: (7,  9),  66: (7, 10),  67: (7, 11),
     68: (7, 12),  69: (7, 13),  70: (7, 14),  71: (7, 15),
    # Actinides — f-block row 8
     90: (8,  2),  91: (8,  3),  92: (8,  4),  93: (8,  5),  94: (8,  6),
     95: (8,  7),  96: (8,  8),  97: (8,  9),  98: (8, 10),  99: (8, 11),
    100: (8, 12), 101: (8, 13), 102: (8, 14), 103: (8, 15),
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_atomic_numbers(dataset_name: str) -> np.ndarray:
    """Return a flat int32 array of atomic numbers from trainset/x[:,0]."""
    path = os.path.join(DIRNAME, dataset_name + ".bp")
    with Stream(path, "r") as f:
        for _ in f.steps():
            z = f.read("trainset/x")[:, 0].astype(np.int32)
            break
    print(f"  {dataset_name}: {len(z):,} atoms")
    return z


def collect_element_frequencies(dataset_list: list[str]) -> dict[int, int]:
    """Read all datasets and return {atomic_number: total_count}."""
    element_frequencies: dict[int, int] = {}
    for name in tqdm(dataset_list, desc="Dataset list"):
        z_arr = read_atomic_numbers(name)
        values, counts = np.unique(z_arr, return_counts=True)
        for z, c in zip(values, counts):
            element_frequencies[z] = element_frequencies.get(z, 0) + int(c)
    return element_frequencies


def build_heatmap(element_frequencies: dict[int, int]) -> np.ndarray:
    """Map element frequencies onto the 2-D periodic table grid."""
    max_row = max(r for r, _ in PERIODIC_TABLE_LAYOUT.values()) + 1
    max_col = max(c for _, c in PERIODIC_TABLE_LAYOUT.values()) + 1
    heatmap = np.zeros((max_row, max_col))
    for z, freq in tqdm(element_frequencies.items(), desc="building heatmap"):
        if z in PERIODIC_TABLE_LAYOUT:
            row, col = PERIODIC_TABLE_LAYOUT[z]
            heatmap[row, col] = freq
    return heatmap


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _annotate_elements(ax: plt.Axes, heatmap: np.ndarray, text_threshold: float) -> None:
    """Draw element borders and Z / symbol labels on *ax*.

    Font sizes are tuned for a 7.16-inch IEEE two-column figure:
      - atomic number: 4pt (superscript-style, top of cell)
      - element symbol: 5pt (bold, centre of cell)
    """
    for z, (row, col) in tqdm(PERIODIC_TABLE_LAYOUT.items(), desc="annotating"):
        rect = patches.Rectangle(
            (col - 0.5, row - 0.5), 1, 1,
            linewidth=0.4, edgecolor="black", facecolor="none",
        )
        ax.add_patch(rect)
        freq = heatmap[row, col]
        symbol = ELEMENT_SYMBOLS[z - 1]
        text_color = "black" if freq < text_threshold else "white"
        # Atomic number — small, upper portion of cell
        ax.text(col, row + 0.22, str(z),
                ha="center", va="center", fontsize=4,
                color=text_color)
        # Element symbol — bold, lower-centre of cell
        ax.text(col, row - 0.18, symbol,
                ha="center", va="center", fontsize=5, fontweight="bold",
                color=text_color)


def plot_heatmap_linear(heatmap: np.ndarray, n_datasets: int) -> None:
    """Save a linear-scale periodic table heatmap sized for IEEE two-column."""
    cmap = ListedColormap(["white"] + plt.cm.YlGnBu(np.linspace(0, 1, 256)).tolist())

    # 7.16 in = full two-column IEEE width; height scaled to match aspect ratio
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    im = ax.imshow(heatmap, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1e10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    _annotate_elements(ax, heatmap, text_threshold=5e9)

    ax.set_title("Periodic Table Heatmap of Element Frequencies", fontsize=9, pad=4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Frequency", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out = f"periodic_table_heatmap_{n_datasets}sets.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


def plot_heatmap_log(heatmap: np.ndarray, n_datasets: int) -> None:
    """Save a log-scale periodic table heatmap sized for IEEE two-column."""
    cmap = ListedColormap(["white"] + plt.cm.YlGnBu(np.linspace(0, 1, 256)).tolist())

    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    im = ax.imshow(
        heatmap, cmap=cmap,
        norm=LogNorm(vmin=1e2, vmax=1e10),
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    _annotate_elements(ax, heatmap, text_threshold=1e6)

    ax.set_title("Periodic Table Heatmap of Element Frequencies", fontsize=9, pad=4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Frequency (log scale)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out = f"periodic_table_heatmap_{n_datasets}sets_log.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Datasets ({len(DATASET_LIST)}):", DATASET_LIST)

    element_frequencies = collect_element_frequencies(DATASET_LIST)

    # Quick coverage summary
    present = sorted(z for z, c in element_frequencies.items() if c > 0)
    symbols = [ELEMENT_SYMBOLS[z - 1] for z in present]
    print(f"\n{len(present)} elements covered: {', '.join(symbols)}")

    heatmap = build_heatmap(element_frequencies)

    present_vals = heatmap[heatmap > 0]
    if present_vals.size:
        print(f"Frequency range: {present_vals.min():.2e} – {present_vals.max():.2e}")

    plot_heatmap_linear(heatmap, n_datasets=len(DATASET_LIST))
    plot_heatmap_log(heatmap, n_datasets=len(DATASET_LIST))


if __name__ == "__main__":
    main()
