"""
composition.py - synthetic molecular density composition for different tissue types.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from scipy import ndimage

from msisim.tissue_model import Tissue, BACKGROUND, CYTOPLASM, MEMBRANE, NUCLEUS


@dataclass
class Composition:
    """Maps (cell type, compartment) -> density for each species.

    type_means          (S, T)  mean density of species s in cell type t
    compartment_weights (S, 3)  multipliers for (cytoplasm, membrane, nucleus)
    cell_cv             (S,)    per-cell biological variability (log-normal, so
                                densities stay positive)
    background          (S,)    off-tissue density (e.g. delocalised analyte)
    """
    species: Sequence[str]
    type_means: np.ndarray
    compartment_weights: Optional[np.ndarray] = None
    cell_cv: float | np.ndarray = 0.1
    background: float | np.ndarray = 0.0

    def render(self, tissue: Tissue, cell_types: np.ndarray, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        tm = np.atleast_2d(np.asarray(self.type_means, float))
        S = tm.shape[0]
        cw = np.ones((S, 3)) if self.compartment_weights is None else np.asarray(self.compartment_weights, float)
        cv = np.broadcast_to(np.asarray(self.cell_cv, float), (S,))
        bg = np.broadcast_to(np.asarray(self.background, float), (S,))

        sig = np.sqrt(np.log1p(cv ** 2))[:, None]
        factors = np.exp(rng.normal(-0.5 * sig ** 2, sig, (S, tissue.n_cells)))  # mean 1
        per_cell = tm[:, cell_types] * factors                                   # (S, n_cells)

        on = tissue.labels >= 0
        lab, comp = tissue.labels[on], tissue.compartments[on].astype(int) - 1
        dens = np.empty((S,) + tissue.grid.shape)
        for s in range(S):
            dens[s] = bg[s]
            dens[s][on] = per_cell[s, lab] * cw[s, comp]
        return dens



#defining different molecular/chemical compositions


def types_stripes(tissue: Tissue, period_um: float, n_types: int = 2, axis: str = "x") -> np.ndarray:
    """
    Tissue stripes
    """
    coord = tissue.nuclei[:, 0 if axis == "x" else 1]
    return np.floor(coord / period_um).astype(int) % n_types


def types_checkerboard(tissue: Tissue, period_um: float, n_types: int = 2) -> np.ndarray:
    """
    Checkerboard pattern.
    """
    ix = np.floor(tissue.nuclei[:, 0] / period_um).astype(int)
    iy = np.floor(tissue.nuclei[:, 1] / period_um).astype(int)
    return (ix + iy) % n_types


def types_random_field(tissue: Tissue, correlation_um: float, proportions: Sequence[float],
                       seed: int = 0) -> np.ndarray:
    """Smooth Gaussian random field thresholded at quantiles -> contiguous
    tissue domains with given type proportions (e.g. tumour / stroma)."""
    rng = np.random.default_rng(seed)
    g = tissue.grid
    field = ndimage.gaussian_filter(rng.standard_normal(g.shape), correlation_um / g.dx)
    r = np.clip(np.round(tissue.nuclei[:, 1] / g.dx).astype(int), 0, g.ny - 1)
    c = np.clip(np.round(tissue.nuclei[:, 0] / g.dx).astype(int), 0, g.nx - 1)
    vals = field[r, c]
    p = np.asarray(proportions, float)
    edges = np.quantile(vals, np.cumsum(p)[:-1] / p.sum())
    return np.searchsorted(edges, vals)