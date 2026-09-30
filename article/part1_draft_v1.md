<!--
Part 1 of a series. Target: about 10 minutes. Primary keyword: physics-informed
neural network / PINN PyTorch Navier-Stokes inverse problem. Secondary: PINN
inverse problem from scratch, wall shear stress neural network, Navier-Stokes
PINN tutorial, learnable parameter viscosity, stenosis blood flow PINN.

Series plan (source material for the cut sections is in part1_long_version.md):
  Part 1 (this): residual, learnable viscosity, results, the cancelling-errors finding.
  Later: the from-scratch CFD solver and the reference-calculation bug; sensor
  placement; the loss-weight story; the pulsatile cardiac cycle (draft.md).

Publishing metadata
  SEO title (<=60): How to Use a PINN for a Navier-Stokes Inverse Problem
  Meta description (<=155): Build a PyTorch PINN that recovers a Navier-Stokes
    blood flow field, unknown viscosity and wall shear stress from 20 noisy sensors.
  Tags: Physics Informed Neural Nets, PyTorch, Deep Learning,
    Computational Fluid Dynamics, Machine Learning
  Hero image: figures/phase1_hero.png (no text, about 2.3:1)

Voice + tell-checklist: writing-guidelines.md. Every number traces to
dev-log.md or script output (scripts/evaluate.py, scripts/make_article_figures.py).
-->

# How to Use a PINN for a Navier-Stokes Inverse Problem

### A from-scratch PyTorch build that recovers blood flow and an unknown viscosity in a stenosed artery from 20 noisy sensors

![PINN-reconstructed blood flow around an arterial narrowing, with sensor locations and the wall coloured by inferred wall shear stress](figures/phase1_hero.png)
*The PINN's velocity field around the stenosis, cropped to the throat. Red rings are the sensors in view, and the wall glows where the PINN infers the most shear stress. This is model output, not ground truth.*

I gave a physics-informed neural network (PINN) twenty noisy velocity readings from a narrowed artery and asked for two things nobody measured: the whole flow field and the viscosity of the fluid. The field came back within 4.7%. The viscosity came back 21.8% high. Wall shear stress, the quantity the exercise was for, looked like it matched the CFD reference, and it took me a while to see why that was wrong.
~ I do not like this, it does not fit here, and it sounds so AI. Specially the "nobody measured", "took me a while to see why that was wrong", and the style of it as a whole 


This is the first article in a series on building a PINN for blood flow in PyTorch, with no DeepXDE or other PINN library. It covers the core: writing the Navier-Stokes residual, making an unknown viscosity learnable, and reading the results without fooling yourself. Later articles cover the CFD solver that produced the ground truth and the pulsatile cardiac cycle.

> What you'll get: a Navier-Stokes residual in about fifty lines of plain PyTorch, a learnable viscosity that was 9.1% off at epoch 8,200 and 21.8% off at epoch 15,000 with nothing in the loss curves to say which one to trust, and a wall shear stress error of 24.4% that hid two larger errors pointing in opposite directions.
~ too many numbers for a takeaway, and I do not even think we need a takeaway


Wall shear stress is the frictional drag of blood along a vessel wall, and it is close to unmeasurable in a clinic. ~"The most used techniques, like" Doppler ultrasound loses accuracy near a boundary, and 4D-flow MRI has to differentiate a velocity field sampled at millimetre-scale voxels [1]. Reported values come in low against CFD by factors of three to six, so it usually gets inferred from a patient-specific CFD model. I wanted the inverse route: reconstruct the field from sparse noisy point measurements, the way a diagnostic workflow would have them.
~ What does the [1] refer to? Put it or drop it

## Writing the Navier-Stokes residual in PyTorch

A PINN is an MLP that maps coordinates to physical quantities. Here, `(x, y) -> (u, v, p)`: two velocity components and pressure. Alongside a supervised term fitting whatever measurements exist, the loss gets a term built from the PDE's *residual*: how badly the network's own output violates the equation, obtained by differentiating the network with respect to its inputs. For incompressible Navier-Stokes that means continuity plus one momentum equation per velocity component.

```python
def _grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
        allow_unused=True, materialize_grads=True,
    )[0]

def ns_residuals(model, x, y, nu):
    x, y = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    u, v, p = model(x, y)

    u_x, u_y = _grad(u, x), _grad(u, y)
    v_x, v_y = _grad(v, x), _grad(v, y)
    p_x, p_y = _grad(p, x), _grad(p, y)
    u_xx, u_yy = _grad(u_x, x), _grad(u_y, y)
    v_xx, v_yy = _grad(v_x, x), _grad(v_y, y)

    continuity = u_x + v_y
    momentum_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, momentum_u, momentum_v
```

Drive those three toward zero at points scattered across the domain and the network stops interpolating your twenty data points. It becomes closer to a solution of the PDE that also agrees with your measurements. The flag that matters is `create_graph=True`, which lets you differentiate a derivative: the viscous term needs `u_xx`, and without it the second call raises on the first epoch.
~ "The flag that matters" reads as AI


## Making viscosity a learnable parameter

In the forward problem the physics is fully specified and you want the field. In the inverse case you have scattered noisy measurements and something unknown about the system, and you want the field *and* the missing piece. Garay et al. infer reduced-order model parameters alongside velocity fields from noisy aortic data [2], Arzani et al. recover near-wall blood flow from sparse measurements [3], and Satyadharma et al. study how PINN performance holds up with sparse noisy velocity data [4].
~I don't really like adding references like that, just make it about mine, or be a bit vague

My unknown is viscosity. It isn't a constant passed into `ns_residuals`. It's a parameter optimized jointly with the network weights, held in log space so it stays positive:

```python
log_nu = torch.nn.Parameter(torch.tensor(float(np.log(4.0 * nu_true_hint))))
```

Log space matters here. The nondimensional viscosity is `0.005`, two and a half orders of magnitude below the scale the network weights live at, and Adam's step sizes are tuned for the latter. Optimizing `log_nu` puts the parameter on a scale where a step means something.
~ "matters here" "where it means something" are AI tells

The `4.0` starts it at four times the true value, since a parameter that starts on the answer is a very expensive confirmation of something you already knew. A factor of four is still an anchor. The estimate approaches the true value from that side and never crosses it, and a literature value for blood would be a more defensible starting point.
~This sounds weird. And why not start from the literature then???

## The setup: a stenosed artery and twenty sensors
~ Just "the setup" would be fine

The domain is a 2D channel of height `H` and length `5H`, narrowed in the middle by a one-sided Gaussian bump that gives a 50% diameter stenosis. The physics is incompressible Navier-Stokes at `Re = 200`, nondimensionalized by channel height and inlet velocity, which is about `0.11 m/s` in a 6 mm carotid, the slow end of physiological. The `nu` in the residual is *kinematic* viscosity, roughly `3.3 × 10⁻⁶ m²/s` for blood (dynamic `3.5 × 10⁻³ Pa·s` over a density of `1060 kg/m³`).

Ground truth comes from a Chorin projection solver I wrote from scratch (640 by 128 cells, immersed-boundary masking for the wall). Its Poiseuille profile error is `5.9e-5`, mass is conserved within `0.1%` across stations, and divergence sits at machine precision. The PINN never sees any of it.
~ "The PINN never sees any of it" AI

What it does see is twenty points drawn uniformly from the fluid interior, each with a velocity vector plus Gaussian noise. No pressure readings. The noise is 7% of the *mean inlet* velocity, not of the local speed, so at these probes it runs from 2.8% in the fastest flow to 56.6% at one that landed in nearly stagnant fluid (median 4.6%). Exactly one of the twenty sits within `0.5H` of the throat.

The network is six layers of 32 `tanh` units, about 5,500 parameters. It uses `tanh` because the momentum residual needs `u_xx`, and ReLU's second derivative is zero almost everywhere. The loss has three weighted groups:

| Group | Weight | What it contains |
|---|---|---|
| `data` | 100 | MSE against the twenty readings |
| `pde` | 1 | MSE of the three residuals at 1,500 collocation points, resampled every 200 epochs |
| `bc` | 10 | no-slip on both walls (100 points each), the exact inlet profile (100 points), one point pinning `p = 0` |

Pressure appears in the equations only as a gradient, so without that pinned point nothing stops the network from settling on `p + 400`. There is deliberately no outlet condition, on the reasoning that over-specifying an outflow is a good way to get a wrong answer confidently. Training is Adam for 15,000 epochs, then 300 iterations of L-BFGS.

## The viscosity was right at epoch 8,200 and wrong at the end

Jointly optimizing a physical parameter with network weights sounds like it should be free: one more tensor in the optimizer. It wasn't. Viscosity is only identifiable once the velocity field is roughly right, and a single scalar converges on a completely different timescale than a field does.

The first run put everything on one learning rate. Viscosity passed within 7.6% of the truth around epoch 1,200 and kept going, finishing at 0.00186 against a true 0.005, a 63% error, while the data and boundary losses looked fine throughout.

A parameter that sails through the right answer and out the other side is moving too fast relative to whatever constrains it. So I gave `log_nu` its own learning rate at 0.1× the network's, and froze it during L-BFGS. That run came in at 124% error. It approached the truth monotonically and did not arrive before training stopped. The failure mode had flipped, which put the answer somewhere between the two.

At 0.3×, with epochs raised to 15,000, it came down from above and got close without ever crossing: the best estimate was 0.00546 at epoch 8,200, a 9.1% error. Then it drifted back up and settled at **0.00609 against a true 0.005, a 21.8% error**.

```python
optimizer = torch.optim.Adam([
    {"params": model.parameters(), "lr": cfg["adam_lr"]},
    {"params": [log_nu], "lr": cfg["adam_lr"] * 0.3},
])
```

![Per-term training losses and the viscosity trajectory normalized by the true value](figures/phase1_training_curves.png)
*Top: unweighted per-term losses. Bottom: recovered viscosity divided by the true value. It approaches from above, is closest near epoch 8,200 at 9.1%, and drifts back up to 21.8%. The loss curves give no sign of the drift.*

The parameter was twice as accurate two thirds of the way through training as it was at the end, and nothing in the loss curves marks epoch 8,200 as special. If you recover a physical parameter this way, the final-epoch value is not obviously the one to report, and I have no principled stopping criterion to offer. Checkpoint the trajectory and look at it. The run was also budget limited: the PDE term was still falling when Adam stopped, about 140 times above the data term.

~This whole section is written in such a boring way, it needs to sort of be unified into an explanation, not moment by moment what was happening

## Results: a 4.7% field error and a wall shear stress number I almost trusted
~ Again, just "Results" is probably fine

The field reconstruction comes in at **4.7% relative L2 error** on both `u` and velocity magnitude. The `v` component alone sits at 29%, because channel flow is overwhelmingly axial and the same absolute error divides by a much smaller number.

![Ground truth versus PINN velocity magnitude with sensors on both panels and the zero-u contour in red](figures/phase1_field_comparison.png)
*Ground-truth CFD (top) against the PINN (bottom), with the twenty sensors on both. The red line is the `u = 0` contour: flow runs backwards below it. The PINN reproduces the reversed-flow region along the lower wall, plus a short extra fragment on the downstream flank of the bump.*

Pressure is inferred too. After removing each field's mean (only its gradient appears in the equations), the error is 12.8%, and the pressure *drop* from end to end comes out at 3.30 against a true 3.16, about 4% high. That is the kind of quantity fractional flow reserve is built on, and no sensor observed it.

Wall shear stress, the quantity no sensor touched, comes back at **24.4% relative L2 error**. The PINN's peak looks well matched, 0.316 against 0.335, and when I went to explain why the error was still 24.4%, I found I had been fooling myself.

Wall shear stress here is `nu` times a velocity gradient, and *both* factors are inferred. My evaluation compared the PINN's recovered viscosity times the PINN's gradient against the true viscosity times the true gradient. Separate them and the picture changes. The recovered viscosity is 21.8% high. The network's near-wall gradient is 22% low across the whole wall, and at the station where the true gradient peaks it reads 0.766 of the truth. The two errors point in opposite directions and largely cancel in the product. Feed the true viscosity into the same network and the peak drops to 0.259.

![Three panels: wall velocity gradient, viscosity, and wall shear stress with the true-viscosity curve](figures/phase1_cancelling_errors.png)
*Gradient times viscosity equals wall shear stress. The PINN's gradient is low (1), its viscosity is high (2), and the product looks fine (3, dashed). The dotted curve is the same network with the true viscosity, which is what the gradient alone gives you.*

So there is a systematic underestimate, and it was hiding inside a quantity that looked fine. If you take one thing from this article, take this: a product of two inferred quantities can agree with the truth while neither factor does. Report the factors.

Two caveats. Both curves use `nu du/dy`, which is the true wall shear only where the wall is horizontal, so on the inclined flanks it is an approximation. And the 24.4% was 97.8% before I fixed a bug in my reference calculation, which is a story for the article on the solver.

~ Nice, but I would appreciate more if the end result was that the PINN  did end uo working. Furthermore, the plot of true wall-shear stress is extremely spikey... and both curves math it well, just one above the spikes and one below...

## Before you build your own inverse PINN

Give any learnable physical parameter its own learning rate, and expect to tune it separately from the network's. The coupling between parameter and field is the whole difficulty, and it doesn't announce itself in the loss curves. All three of my viscosity runs had healthy-looking data and boundary losses, so watch the per-term losses and save the parameter's trajectory.

Write a manufactured-solution test for the residual code. It costs about twenty lines: pick something polynomial, derive the residuals by hand, assert they match. Mine caught an autograd edge case before it reached a training run.

## Where this breaks

The domain is too short. Behind the bump the flow separates and runs backwards along the wall, and in a 5H channel that reversed region never closes. It is still going at the outlet, where the first few fluid cells read `-0.004` to `-0.054`. An outflow condition assumes fluid is leaving, so downstream of about `x = 2.2` the boundary, not the physics, is steering the reference solution. The shear stress peak at `x ≈ 1.9` sits upstream of that.

Blood is also shear-thinning, so a single constant viscosity is weakest in the separated zone behind the stenosis, the same place the domain is too short.

The network gets exact boundary data as well (the inlet profile, the no-slip walls, the geometry), so I checked what the sensors add. I trained the same setup with the data weight at zero and the viscosity fixed at its true value. That run reached 26.4% field error and 52.9% wall shear stress error, against 4.7% and 24.4% with the sensors, and its near-wall gradient at the peak was 0.494 of the truth instead of 0.766. It is one run with one seed, but the boundary data and the physics do not pin the field down on their own. Twenty noisy probes carry a large share of the answer.

The flow is 2D, the walls are rigid, and the noise is additive Gaussian, which stands in for Doppler without simulating beam angle or speckle. The link between wall shear stress and plaque location is well studied but contested [5].

The steady case gave me a field I could trust to 5% and a wall shear stress I could not, until I split it into its two factors. The next articles turn the flow pulsatile, where the same damping comes back much harder: the PINN's peak reaches 27% of the true peak. It took three fixes and two failed experiments to get anywhere. If that is the story you want, follow along.

If you have recovered a physical parameter with a PINN, I'd like to know whether your best epoch was your last.

All code, configs, and figures: [repo link].

---

~Good the blood non constant viscosity. Too many numbers. "I could trust" sounds AI.

A few comments in general:
- Too many numbers, I don't want that many percentage errors, it is fine to be a bit more handwavy with that, it fits the article vibe better "high" "low" "within the injected noise" etc etc
- The fact that the end product is not a good PINN is dissapointing. I watched the gif, and we do not capture any of the vorticity downstream either.
- The style is still too AI, I hope you take some style guides away from this review.

## References

[1] P. Dyverfeldt, M. Bissell, A.J. Barker, et al., "4D flow
cardiovascular magnetic resonance consensus statement" (2015), *Journal
of Cardiovascular Magnetic Resonance*, 17:72.

[2] J. Garay, M. Dunstan, S. Uribe, F. Sahli Costabal, "Physics-informed
neural networks for parameter estimation in blood flow models" (2024),
*Computers in Biology and Medicine*, 178:108706.

[3] A. Arzani, J.-X. Wang, R.M. D'Souza, "Uncovering near-wall blood flow
from sparse data with physics-informed neural networks" (2021), *Physics
of Fluids*, 33:071905.

[4] A. Satyadharma, M.-J. Chern, H.-C. Kan, Harinaldi, J. Julian,
"Assessing physics-informed neural network performance with sparse noisy
velocity data" (2024), *Physics of Fluids*, 36(10):103619.

[5] P.D. Weinberg, "Haemodynamic Wall Shear Stress, Endothelial
Permeability and Atherosclerosis: A Triad of Controversy" (2022),
*Frontiers in Bioengineering and Biotechnology*, 10:836680.
