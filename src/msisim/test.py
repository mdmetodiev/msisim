"""
msi_sim.py - synthetic tissue generation + raster-mode MSI image formation.

Conventions
-----------
* Lengths in micrometres, times in seconds.
* Arrays are indexed [row, col] = [y, x];  x = col * dx,  y = row * dx.
* Densities have shape (n_species, ny, nx), units: amount per um^2.

Physics (see SI, eqs S5-S9)
---------------------------
    d rho_s / dt = -k_s * I(x - x_c(t), y - y_c) * rho_s
with a Gaussian beam I = I0 * exp(-(x^2)/(2 sx^2) - (y^2)/(2 sy^2)) moving at
constant velocity v. While the beam centre moves from x=a to x=b the dose is

    D(x, y) = I0 * g_y(y) * (sx*sqrt(pi/2)/v) * [erf((x-a)/(sqrt2 sx)) - erf((x-b)/(sqrt2 sx))]

so rho -> rho * exp(-k * D) exactly, and the pixel signal is gamma * (mass removed).
Only a local window (+- n_sigma beam widths) changes per pixel, so memory is
O(size of the density map) and nothing per-pixel is stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.special import erf

BACKGROUND, CYTOPLASM, MEMBRANE, NUCLEUS = 0, 1, 2, 3
_SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tissue geometry
# --------------------------------------------------------------------------- #
@dataclass
class Tissue:
    grid: Grid
    nuclei: np.ndarray        # (n_cells, 2) as (x, y) in um
    labels: np.ndarray        # (ny, nx) int32, cell index, -1 = off tissue
    compartments: np.ndarray  # (ny, nx) uint8: BACKGROUND/CYTOPLASM/MEMBRANE/NUCLEUS

    @property
    def n_cells(self) -> int:
        return len(self.nuclei)

    # ---- construction ---------------------------------------------------- #
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

    # ---- helpers --------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Cell-type assignment: each returns an int array of length n_cells
# --------------------------------------------------------------------------- #
def types_stripes(tissue: Tissue, period_um: float, n_types: int = 2, axis: str = "x") -> np.ndarray:
    coord = tissue.nuclei[:, 0 if axis == "x" else 1]
    return np.floor(coord / period_um).astype(int) % n_types


def types_checkerboard(tissue: Tissue, period_um: float, n_types: int = 2) -> np.ndarray:
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


# --------------------------------------------------------------------------- #
# Chemical composition
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GaussianBeam:
    """Normalised Gaussian beam, as in the thesis Eq. 3.4:

        I(x, y) = I0 / (2 pi sx sy) * exp(-x^2/2sx^2) * exp(-y^2/2sy^2)

    so I0 is the *total* beam power (the double integral of I), not the peak.
    This is the physically meaningful convention: shrinking the beam at fixed
    laser power concentrates the same power into a smaller spot.
    """
    sigma_x: float     # um
    sigma_y: float     # um
    I0: float = 1.0    # total beam power; peak intensity = I0 / (2 pi sx sy)

    @classmethod
    def from_peak(cls, sigma_x: float, sigma_y: float, peak: float) -> "GaussianBeam":
        return cls(sigma_x, sigma_y, peak * 2 * np.pi * sigma_x * sigma_y)

    @property
    def peak(self) -> float:
        return self.I0 / (2 * np.pi * self.sigma_x * self.sigma_y)

    def fwhm(self) -> tuple[float, float]:
        f = 2 * np.sqrt(2 * np.log(2))
        return f * self.sigma_x, f * self.sigma_y


@dataclass(frozen=True)
class RasterScan:
    """`pixel_x` / `pixel_y` are *sampling intervals* (pixel pitch), not areas.
    Along a line the spectrometer takes one spectrum every t_sampling seconds
    while the stage moves at `velocity`, so pixel_x = velocity * t_sampling.
    `pixel_y` is the line pitch and is independent of velocity (v_y = 0).
    """
    x_start: float
    y_start: float
    n_pix: int              # spectra per line
    n_lines: int
    pixel_x: float          # um; = velocity * t_sampling
    pixel_y: float          # um; line pitch
    velocity: float         # um/s
    bidirectional: bool = False
    duty_cycle: float = 1.0  # fraction of each sampling interval actually acquired

    @classmethod
    def from_timing(cls, x_start: float, y_start: float, n_pix: int, n_lines: int,
                    velocity: float, t_sampling: float, pixel_y: float,
                    bidirectional: bool = False, duty_cycle: float = 1.0) -> "RasterScan":
        return cls(x_start, y_start, n_pix, n_lines, velocity * t_sampling,
                   pixel_y, velocity, bidirectional, duty_cycle)

    @property
    def t_sampling(self) -> float:
        return self.pixel_x / self.velocity


def scan_extent(scan: "RasterScan") -> tuple[float, float, float, float]:
    """imshow extent (um) for an acquired image, so it overlays the density map.
    x: the pixel value is the flux integrated while the beam crossed
       [x_start + j*pixel_x, x_start + (j+1)*pixel_x], so its centre is half a
       pixel to the right of the nominal position.
    y: no integration happens in y - line i sits exactly at y_start + i*pixel_y.
    """
    return (scan.x_start,
            scan.x_start + scan.n_pix * scan.pixel_x,
            scan.y_start + (scan.n_lines - 0.5) * scan.pixel_y,
            scan.y_start - 0.5 * scan.pixel_y)


def single_pass_peak_dose(beam: GaussianBeam, scan: RasterScan) -> float:
    """Dose (integral of I dt) on the line axis for one complete pass."""
    return beam.peak * beam.sigma_x * np.sqrt(2 * np.pi) / scan.velocity


def k_for_consumption(fraction: float, beam: GaussianBeam, scan: RasterScan) -> float:
    """Desorption coefficient that removes `fraction` of material on the line
    axis in one isolated pass. fraction -> 1: MALDI-like; small: SIMS/DESI-like.
    This is the one dimensionless number that actually controls the physics."""
    return -np.log1p(-fraction) / single_pass_peak_dose(beam, scan)


def simulate_raster(density: np.ndarray, grid: Grid, beam: GaussianBeam, scan: RasterScan,
                    k: float | Sequence[float], *, ionisation: float | Sequence[float] = 1.0,
                    n_sigma: float = 5.0, inplace: bool = False,
                    callback: Optional[Callable[[int, int, np.ndarray], None]] = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Raster-mode image formation.

    Returns (image, depleted_density). image has shape (S, n_lines, n_pix)
    (or (n_lines, n_pix) for a 2-D input) and holds gamma * mass desorbed
    while the beam centre was inside each pixel.
    `callback(line, pixel, rho)` is called after every pixel - use it to render
    animation frames without storing the history.
    """
    squeeze = density.ndim == 2
    rho = density if inplace else density.copy()
    rho = rho[None] if squeeze else rho
    S = rho.shape[0]
    rate = (np.broadcast_to(np.asarray(k, float), (S,)) * beam.peak)[:, None, None]
    gamma = np.broadcast_to(np.asarray(ionisation, float), (S,))

    x, y, dx = grid.x, grid.y, grid.dx
    sx, sy = beam.sigma_x, beam.sigma_y
    const = sx * np.sqrt(np.pi / 2) / scan.velocity    # the 1/v of Eq. 2.20 lives here
    img = np.zeros((S, scan.n_lines, scan.n_pix))
    duty = float(np.clip(scan.duty_cycle, 0.0, 1.0))

    def sweep(a: float, b: float, r0: int, r1: int, gy: np.ndarray) -> np.ndarray:
        """Move the beam centre from a to b (either direction), deplete rho in
        place, return the mass removed per species."""
        lo, hi = (a, b) if a <= b else (b, a)
        if hi <= lo:
            return np.zeros(S)
        c0 = max(0, int(np.floor((lo - n_sigma * sx) / dx)))
        c1 = min(grid.nx, int(np.ceil((hi + n_sigma * sx) / dx)) + 1)
        if c0 >= c1:
            return np.zeros(S)
        xs = x[c0:c1]
        # integral of the beam over the time its centre travels lo -> hi
        dose_x = const * (erf((xs - lo) / (_SQRT2 * sx)) - erf((xs - hi) / (_SQRT2 * sx)))
        expo = rate * (gy * dose_x[None, :])[None]                         # (S, h, w)
        win = rho[:, r0:r1, c0:c1]                                         # view
        removed = win * -np.expm1(-expo)
        win -= removed
        return removed.sum(axis=(1, 2)) * dx * dx

    for i in range(scan.n_lines):
        yc = scan.y_start + i * scan.pixel_y
        r0 = max(0, int(np.floor((yc - n_sigma * sy) / dx)))
        r1 = min(grid.ny, int(np.ceil((yc + n_sigma * sy) / dx)) + 1)
        if r0 >= r1:
            continue
        gy = np.exp(-0.5 * ((y[r0:r1] - yc) / sy) ** 2)[:, None]           # (h, 1)
        reverse = scan.bidirectional and (i % 2 == 1)
        order = range(scan.n_pix - 1, -1, -1) if reverse else range(scan.n_pix)

        for j in order:
            lo = scan.x_start + j * scan.pixel_x
            hi = lo + scan.pixel_x
            if reverse:                       # beam enters this pixel at hi, leaves at lo
                split = hi - duty * scan.pixel_x
                img[:, i, j] = sweep(hi, split, r0, r1, gy)
                sweep(split, lo, r0, r1, gy)  # dead time: material removed, no signal
            else:
                split = lo + duty * scan.pixel_x
                img[:, i, j] = sweep(lo, split, r0, r1, gy)
                sweep(split, hi, r0, r1, gy)
            if callback is not None:
                callback(i, j, rho)

    img *= gamma[:, None, None]
    return (img[0], rho[0]) if squeeze else (img, rho)


def detect(signal: np.ndarray, gain: float = 1.0, *, poisson: bool = True,
           multiplicative_cv: float = 0.0, additive_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    """Detector model: optional shot-to-shot log-normal fluctuation (matrix
    crystals, laser energy), Poisson ion counting, additive electronic noise.
    Poisson gives the intensity-noise correlation seen in real MSI data."""
    rng = np.random.default_rng(seed)
    mu = signal * gain
    if multiplicative_cv > 0:
        s = np.sqrt(np.log1p(multiplicative_cv ** 2))
        mu = mu * np.exp(rng.normal(-0.5 * s * s, s, mu.shape))
    out = rng.poisson(np.clip(mu, 0, None)).astype(float) if poisson else mu
    if additive_sigma > 0:
        out = out + rng.normal(0.0, additive_sigma, out.shape)
    return out


# --------------------------------------------------------------------------- #
# Self-tests / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from time import perf_counter

    grid = Grid.from_extent(600, 600, dx=1.0)
    tissue = Tissue.jittered_lattice(grid, (12, 12), jitter=0.12, lloyd_iters=2, seed=16)
    types = types_stripes(tissue, period_um=60)
    comp = Composition(
        species=["PC 34:1", "PC 38:4"],
        type_means=[[1.0, 0.3],        # species 0 in type 0, type 1
                    [0.2, 0.8]],       # species 1
        compartment_weights=[[1.0, 2.0, 0.3],
                             [1.0, 1.5, 0.3]],
        cell_cv=0.15,
    )
    rho0 = comp.render(tissue, types, seed=1)

    beam = GaussianBeam(sigma_x=15, sigma_y=15, I0=1e4)      # I0 = total power
    scan = RasterScan.from_timing(x_start=50, y_start=50, n_pix=20, n_lines=20,
                                  velocity=50, t_sampling=0.5, pixel_y=25)
    print(f"pixel_x = v * t_sampling = {scan.pixel_x} um, beam FWHM = "
          f"{beam.fwhm()[0]:.1f} x {beam.fwhm()[1]:.1f} um")
    k = k_for_consumption(0.99, beam, scan)

    t = perf_counter()
    img, rho = simulate_raster(rho0, grid, beam, scan, k=[k, 0.3 * k])
    print(f"{scan.n_lines * scan.n_pix} pixels x {rho0.shape[0]} species in {perf_counter() - t:.2f} s")

    # 1) mass balance: signal + remaining == initial
    err = (img.sum(axis=(1, 2)) + rho.sum(axis=(1, 2)) - rho0.sum(axis=(1, 2))) / rho0.sum(axis=(1, 2))
    print("relative mass-balance error per species:", err)

    # 2) one full line vs the closed form of Eq. 2.20 / 3.5 (y-factor inside the exponent)
    one = RasterScan(x_start=100, y_start=300, n_pix=16, n_lines=1,
                     pixel_x=25, pixel_y=25, velocity=50)
    _, rho_line = simulate_raster(rho0[0], grid, beam, one, k=k)
    Y, X = grid.mesh()
    a, b = one.x_start, one.x_start + one.n_pix * one.pixel_x
    cc = beam.sigma_x * np.sqrt(np.pi / 2) / one.velocity
    expo = k * beam.peak * cc \
        * (erf((X - a) / (_SQRT2 * beam.sigma_x)) - erf((X - b) / (_SQRT2 * beam.sigma_x))) \
        * np.exp(-0.5 * ((Y - one.y_start) / beam.sigma_y) ** 2)
    ref = rho0[0] * np.exp(-expo)
    print("max |incremental - closed form| / max rho:", np.abs(rho_line - ref).max() / rho0[0].max())

    # 3) forward vs bidirectional rastering leaves the same total mass removed
    fwd = RasterScan(50, 50, 20, 20, 25, 25, 50)
    bi = RasterScan(50, 50, 20, 20, 25, 25, 50, bidirectional=True)
    i1, r1 = simulate_raster(rho0[0], grid, beam, fwd, k=k)
    i2, r2 = simulate_raster(rho0[0], grid, beam, bi, k=k)
    print("bidirectional: total signal ratio =", i2.sum() / i1.sum(),
          "| max per-pixel difference =", np.abs(i1 - i2).max() / i1.max())

    # 4) duty cycle < 1: material is still consumed, part of the signal is lost
    duty = RasterScan(50, 50, 20, 20, 25, 25, 50, duty_cycle=0.5)
    i3, r3 = simulate_raster(rho0[0], grid, beam, duty, k=k)
    print("duty 0.5: signal kept =", i3.sum() / i1.sum(),
          "| density identical to duty 1:", np.allclose(r3, r1))