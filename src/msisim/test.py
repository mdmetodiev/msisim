from msisim.tissue_model import Tissue, Grid
from msisim.composition import Composition, types_stripes
from msisim.image_formation import GaussianBeam, RasterScan, simulate_raster, k_for_consumption, _SQRT2

import numpy as np
from scipy.special import erf

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