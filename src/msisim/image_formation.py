"""
image_formation.py - raster-mode MSI image formation.

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
from scipy.special import erf

from msisim.tissue_model import Grid, _SQRT2


@dataclass(frozen=True)
class GaussianBeam:
    """Normalised Gaussian beam, as in my thesis Eq. 3.4:

        I(x, y) = I0 / (2 pi sx sy) * exp(-x^2/2sx^2) * exp(-y^2/2sy^2)

    so I0 is the *total* beam power (the double integral of I), not the peak.
    Note that this is a physically meaningful convention: shrinking the beam at fixed
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
    """`pixel_x` / `pixel_y` are *sampling intervals* (pixel pitch), not areas (or "pixel sizes" as commonly referred).
    Along a line the spectrometer takes one spectrum every t_sampling seconds
    while the stage moves at `velocity`, so pixel_x = velocity * t_sampling.
    `pixel_y` is the line pitch and is independent of velocity (v_y = 0), but we can generalise should we need to.
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
