"""Tests for the Phase 2 pulsatile inlet waveform (M9)."""

import numpy as np

from pinn_bloodflow.solver.waveform import cardiac_pulse


def test_waveform_is_periodic():
    t = np.linspace(0, 3.0, 500)  # three full periods (T=1.0)
    f = cardiac_pulse(t, period=1.0)
    f_shifted = cardiac_pulse(t + 1.0, period=1.0)
    assert np.allclose(f, f_shifted, atol=1e-10)


def test_waveform_stays_positive():
    # No momentary inlet flow reversal -- required for the Dirichlet inlet
    # BC to stay physically simple (see waveform.py docstring).
    t = np.linspace(0, 1.0, 2000, endpoint=False)
    f = cardiac_pulse(t, period=1.0)
    assert f.min() > 0.0


def test_waveform_peak_is_in_first_half_of_cycle():
    # Asymmetric (systole-then-diastole) shape, not a symmetric sine: the
    # peak should land distinctly before the cycle midpoint.
    t = np.linspace(0, 1.0, 2000, endpoint=False)
    f = cardiac_pulse(t, period=1.0)
    t_peak = t[np.argmax(f)]
    assert t_peak < 0.4


def test_waveform_mean_reduces_to_baseline():
    # At the default amplitudes, the cycle-averaged multiplier should sit
    # close to the "mean" parameter (the two harmonics have zero mean
    # individually), i.e. it's a modulation of, not a departure from, the
    # steady inflow rate.
    t = np.linspace(0, 1.0, 10000, endpoint=False)
    f = cardiac_pulse(t, period=1.0, mean=1.0)
    assert abs(f.mean() - 1.0) < 1e-6
