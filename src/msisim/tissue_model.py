"""
tissue_model.py - synthetic tissue/cell generation.

Conventions
-----------
* Lengths in micrometres, times in seconds.
* Arrays are indexed [row, col] = [y, x];  x = col * dx,  y = row * dx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import  Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

BACKGROUND, CYTOPLASM, MEMBRANE, NUCLEUS = 0, 1, 2, 3 #TODO: consider if enum is  better?
_SQRT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class Grid:
    ny: int
    nx: int
    dx: float = 1.0  # um per grid point (must be << beam sigma)

    @classmethod
    def from_extent(cls, width_um: float, height_um: float, dx: float = 1.0) -> "Grid":
        return cls(int(round(height_um / dx)), int(round(width_um / dx)), dx)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx

    @property
    def y(self) -> np.ndarray:
        return np.arange(self.ny) * self.dx

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """imshow extent (left, right, bottom, top) in um, for origin='upper'."""
        return (0.0, self.nx * self.dx, self.ny * self.dx, 0.0)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Y, X) with 'ij' indexing, matching array layout."""
        return np.meshgrid(self.y, self.x, indexing="ij")


@dataclass
class Tissue:
    grid: Grid
    nuclei: np.ndarray        # (n_cells, 2) as (x, y) in um
    labels: np.ndarray        # (ny, nx) int32, cell index, -1 = off tissue
    compartments: np.ndarray  # (ny, nx) uint8: BACKGROUND/CYTOPLASM/MEMBRANE/NUCLEUS

    @property
    def n_cells(self) -> int:
        return len(self.nuclei)

    # construction methods
    @classmethod
    def from_nuclei(cls, grid: Grid, nuclei: np.ndarray, *, max_cell_radius: float,
                    nucleus_radius: float, membrane_width: float = 1.0) -> "Tissue":
        """Rasterise the Voronoi tessellation directly: each grid point belongs to
        its nearest nucleus (that *is* the Voronoi definition), so every pixel is
        assigned exactly once - no polygons, no double-counted edges."""
        Y, X = grid.mesh()
        pts = np.column_stack([X.ravel(), Y.ravel()])
        d, idx = cKDTree(nuclei).query(pts, k=2)
        d1, d2 = d[:, 0], d[:, 1]

        labels = idx[:, 0].astype(np.int32)
        comp = np.full(labels.shape, CYTOPLASM, np.uint8)
        # (d2 - d1)/2 ~ distance to the shared Voronoi edge
        comp[0.5 * (d2 - d1) < 0.5 * membrane_width] = MEMBRANE
        comp[d1 > max_cell_radius - 0.5 * membrane_width] = MEMBRANE  # outer tissue border
        comp[d1 < nucleus_radius] = NUCLEUS
        off = d1 > max_cell_radius
        labels[off] = -1
        comp[off] = BACKGROUND
        return cls(grid, np.asarray(nuclei, float),
                   labels.reshape(grid.shape), comp.reshape(grid.shape))

    @classmethod
    def jittered_lattice(cls, grid: Grid, cell_diam: tuple[float, float] = (10.0, 10.0), *,
                         jitter: float = 0.15, margin: Optional[float] = None,
                         lloyd_iters: int = 0, nucleus_frac: float = 0.35,
                         membrane_width: float = 1.0, max_radius_frac: float = 0.8,
                         seed: int = 0) -> "Tissue":
        """Lattice of nuclei with Gaussian jitter given as a *fraction of the cell
        diameter*; optional Lloyd relaxation makes cells more regular/realistic."""
        rng = np.random.default_rng(seed)
        cdx, cdy = cell_diam
        margin = max(cdx, cdy) if margin is None else margin
        xs = np.arange(margin, grid.nx * grid.dx - margin + 1e-9, cdx)
        ys = np.arange(margin, grid.ny * grid.dx - margin + 1e-9, cdy)
        X, Y = np.meshgrid(xs, ys)
        nuclei = np.column_stack([X.ravel(), Y.ravel()])
        nuclei = nuclei + rng.normal(0.0, jitter, nuclei.shape) * np.array([cdx, cdy])

        kw = dict(max_cell_radius=max_radius_frac * max(cdx, cdy),
                  nucleus_radius=0.5 * nucleus_frac * min(cdx, cdy),
                  membrane_width=membrane_width)
        tissue = cls.from_nuclei(grid, nuclei, **kw)
        for _ in range(lloyd_iters):
            tissue = cls.from_nuclei(grid, tissue.centroids(), **kw)
        return tissue

    #helper methods
    def centroids(self) -> np.ndarray:
        Y, X = self.grid.mesh()
        lab = self.labels.ravel()
        m = lab >= 0
        cnt = np.bincount(lab[m], minlength=self.n_cells)
        cx = np.bincount(lab[m], X.ravel()[m], minlength=self.n_cells)
        cy = np.bincount(lab[m], Y.ravel()[m], minlength=self.n_cells)
        c = self.nuclei.copy()
        ok = cnt > 0
        c[ok, 0], c[ok, 1] = cx[ok] / cnt[ok], cy[ok] / cnt[ok]
        return c

    def show(self, values: Optional[np.ndarray] = None, ax=None, cmap="magma"):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        img = self.compartments if values is None else values
        ext = self.grid.extent #(0, self.grid.nx * self.grid.dx, self.grid.ny * self.grid.dx, 0)
        ax.imshow(img, extent=ext, cmap=cmap, interpolation="nearest")
        ax.plot(self.nuclei[:, 0], self.nuclei[:, 1], "w.", ms=1)
        ax.set_xlabel(r"$x$ ($\mu$m)"); ax.set_ylabel(r"$y$ ($\mu$m)")
        return ax