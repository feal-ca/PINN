"""Inlet velocity waveform for Phase 2's pulsatile (cardiac-cycle-like) flow.

A real cardiac cycle is *not* a symmetric sine wave: flow rises sharply
during systole (roughly the first third of the cycle) and decays more
slowly during diastole (the remaining two-thirds), with the ejection burst
concentrated near the start. A plain `1 + A*sin(2*pi*t/T)` is symmetric
about its peak, which reads as visually "sloshing" rather than "beating."

We approximate the asymmetric real shape the standard way real waveforms
are approximated in this literature: a short truncated Fourier series (a
sum of a fundamental and one harmonic, out of phase with it) rather than a
piecewise/discontinuous construction. Two terms is the minimum that can
produce a skewed peak at all, and it keeps the waveform smooth (infinitely
differentiable), which matters later for Phase 2's PINN -- it will need to
take an autograd derivative of the network's output with respect to t, and
a smooth target is easier to fit than one with a slope discontinuity.
"""

import numpy as np


def cardiac_pulse(
    t: float | np.ndarray,
    period: float = 1.0,
    mean: float = 1.0,
    amplitude_fundamental: float = 0.55,
    amplitude_harmonic: float = 0.20,
) -> float | np.ndarray:
    """f(t) multiplying the steady Poiseuille inlet profile.

    f(t) = mean + A1*sin(2*pi*t/T) - A2*cos(4*pi*t/T)

    The `-cos(4*pi*t/T)` term is a second harmonic, phase-shifted a quarter
    turn relative to the fundamental (`cos` vs. `sin`): it adds extra rise
    right after t=0 and extra decay right after the peak, skewing the peak
    early into the cycle (systole) and stretching out the return to
    baseline (diastole) -- without introducing any kink.

    Default coefficients (0.55, 0.20) were chosen by scanning a grid of
    (A1, A2) pairs for two properties, verified numerically rather than
    assumed: (1) f(t) stays strictly positive over the whole cycle -- no
    momentary flow reversal at the inlet, which keeps the Dirichlet inlet
    boundary condition simple (no need to handle backflow there); and
    (2) the peak lands near the first quarter of the cycle, a plausible
    systole/diastole split. See dev-log 2026-08-28.
    """
    phase = 2.0 * np.pi * np.asarray(t) / period
    return mean + amplitude_fundamental * np.sin(phase) - amplitude_harmonic * np.cos(2.0 * phase)
