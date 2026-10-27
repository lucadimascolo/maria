from __future__ import annotations

import healpy as hp
import maria
import maria.cmb.cmb
import numpy as np


def test_get_cmb(tmp_path, monkeypatch):
    # the layout master assumed: I, Q, U and two masks in one table, without standard column names
    npix = hp.nside2npix(32)
    fields = [1e-4 * np.random.default_rng(0).standard_normal(npix) for _ in range(3)] + 2 * [np.ones(npix)]
    path = str(tmp_path / "planck_cmb.fits")
    hp.write_map(path, fields, overwrite=True, dtype=[np.float32] * 5)
    calls = []

    def fake_fetch(url=None, **kwargs):
        calls.append(url)
        return path

    monkeypatch.setattr(maria.cmb.cmb, "fetch", fake_fetch)

    cmb = maria.cmb.get_cmb()
    assert calls == [maria.cmb.cmb.CMB_MAP_SOURCE_URL]
    assert cmb.units == "K_CMB"
    assert cmb.frame.name == "galactic"
    assert np.allclose(np.asarray(cmb.data).reshape(3, -1), np.stack(fields[:3]), rtol=1e-6)

    assert maria.cmb.get_cmb(nside=16).nside == 16

    plan = maria.Plan.generate(
        start_time="2025-01-01T07:00:00",
        scan_type="daisy",
        scan_parameters={"radius": 1.0, "speed": 0.5},
        duration=10,
        sample_rate=10,
        scan_center=(150.0, -30.0),
        frame="ra/dec",
        jitter=0,
    )
    sim = maria.Simulation(
        instrument="test/1deg",
        plans=plan,
        site="cerro_toco",
        cmb="planck",
        cmb_kwargs={"nside": 16},
        noise=False,
        progress_bars=False,
    )
    tod = sim.run(units="uK_CMB")[0]
    assert np.isfinite(tod.signal.compute()).all()
    assert tod.signal.compute().std() > 0


def test_get_cmb_planck_layout(tmp_path, monkeypatch):
    """The Planck file has T (nside 4096) and Q/U (nside 2048) in separate NESTED extensions."""
    from astropy.io import fits

    rng = np.random.default_rng(1)
    T, Q, U = (1e-4 * rng.standard_normal(hp.nside2npix(n)).astype(np.float32) for n in (32, 16, 16))

    def table(columns, nside):
        hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name=name, format="E", unit="K_CMB", array=values) for name, values in columns]
        )
        hdu.header.update({"PIXTYPE": "HEALPIX", "ORDERING": "NESTED", "COORDSYS": "GALACTIC", "NSIDE": nside})
        return hdu

    path = str(tmp_path / "planck.fits")
    fits.HDUList(
        [fits.PrimaryHDU(), table([("TEMPERATURE", T)], 32), table([("Q_POLARIZATION", Q), ("U_POLARIZATION", U)], 16)]
    ).writeto(path)
    monkeypatch.setattr(maria.cmb.cmb, "fetch", lambda **kwargs: path)

    cmb = maria.cmb.get_cmb()
    assert cmb.nside == 16
    data = np.asarray(cmb.data).reshape(3, -1)
    assert np.allclose(data[0], hp.ud_grade(hp.reorder(T, n2r=True), 16), rtol=1e-5)
    assert np.allclose(data[1], hp.reorder(Q, n2r=True))
    assert np.allclose(data[2], hp.reorder(U, n2r=True))


def test_get_cmb_keeps_dipole(tmp_path, monkeypatch):
    """The Planck map keeps its dipole, as on master."""
    nside = 32
    rng = np.random.default_rng(2)
    npix = hp.nside2npix(nside)
    anisotropies = [1e-4 * rng.standard_normal(npix) for _ in range(3)]
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    dipole = 1e-6 + 3.4e-3 * (0.6 * x - 0.8 * z)
    path = str(tmp_path / "planck_cmb.fits")
    hp.write_map(path, [anisotropies[0] + dipole, *anisotropies[1:]], overwrite=True, dtype=[np.float32] * 3)
    monkeypatch.setattr(maria.cmb.cmb, "fetch", lambda **kwargs: path)

    kept = np.asarray(maria.cmb.get_cmb().data).reshape(3, -1)
    assert np.allclose(kept[0], anisotropies[0] + dipole, atol=1e-8)
    assert np.allclose(kept[1:], np.stack(anisotropies[1:]), atol=1e-8)
