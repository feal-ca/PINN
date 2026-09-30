import numpy as np

from pinn_bloodflow.geometry import StenosisParams, build_fluid_mask, is_fluid, stenosis_height, wall_curve_points


def test_max_height_matches_pct_ds():
    params = StenosisParams(H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4)
    x = np.array([params.x0])
    h = stenosis_height(x, params)
    assert np.isclose(h[0], 0.5)


def test_height_decays_away_from_throat():
    params = StenosisParams(H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4)
    x = np.array([params.x0, params.x0 + 5 * params.sigma])
    h = stenosis_height(x, params)
    assert h[0] > h[1]
    assert h[1] < 1e-3


def test_eccentric_mask_blocks_only_bottom():
    params = StenosisParams(H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4, eccentric=True)
    # At the throat, y=0.4 is inside the 0.5-high bump -> solid; y=0.9 is clear -> fluid.
    assert not is_fluid(np.array(params.x0), np.array(0.4), params)
    assert is_fluid(np.array(params.x0), np.array(0.9), params)


def test_symmetric_mask_blocks_both_walls():
    params = StenosisParams(H=1.0, pct_ds=50.0, x0=2.0, sigma=0.4, eccentric=False)
    # h=0.5 at throat -> each wall advances by 0.25, fluid gap is (0.25, 0.75).
    assert not is_fluid(np.array(params.x0), np.array(0.1), params)
    assert not is_fluid(np.array(params.x0), np.array(0.9), params)
    assert is_fluid(np.array(params.x0), np.array(0.5), params)


def test_build_fluid_mask_shape():
    params = StenosisParams()
    x_coords = np.linspace(0, params.Lx, 320)
    y_coords = np.linspace(0, params.H, 64)
    mask = build_fluid_mask(x_coords, y_coords, params)
    assert mask.shape == (64, 320)
    assert mask.dtype == bool
    # y=0 is the physical channel wall itself: always solid, even far
    # upstream of the stenosis where the bump height is ~0.
    assert not mask[0, 0]
    # Everywhere else in that upstream cross-section should be fluid.
    assert mask[1:, 0].all()


def test_wall_curve_points_match_height_function():
    params = StenosisParams(eccentric=True)
    x, y = wall_curve_points(params, n=50)
    assert np.allclose(y, stenosis_height(x, params))
