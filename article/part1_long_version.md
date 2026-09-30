<!--
LONG VERSION of Part 1 (about 3,250 prose words). Kept as source material for later
articles in the series: architecture and loss details, sensor design, the reference-calculation
bug (solver/verification article), error heatmap and raw WSS figures, full limitations.
The publishable Part 1 is part1_draft.md.
-->

<!--
Part 1 of 2. Primary keyword: physics-informed neural network PyTorch
Navier-Stokes inverse problem. Secondary: PINN inverse problem from scratch,
wall shear stress neural network, Navier-Stokes PINN tutorial, learnable
parameter viscosity, stenosis blood flow PINN.
Part 2 covers the from-scratch CFD solver, the pulsatile extension, and the
debugging arc.

Publishing metadata
  SEO title (<=60): PINN in PyTorch: Navier-Stokes Inverse Problem for Blood Flow
  Meta description (<=155): Build a PyTorch PINN that recovers a Navier-Stokes
    blood flow field, unknown viscosity and wall shear stress from 20 noisy sensors.
  Tags: Physics Informed Neural Nets, PyTorch, Deep Learning,
    Computational Fluid Dynamics, Machine Learning
  Hero image: figures/phase1_hero.png (2:1)

Voice + tell-checklist: writing-guidelines.md. Every number traces to
dev-log.md or script output (scripts/evaluate.py, scripts/make_article_figures.py).
-->

# Physics-Informed Neural Network in PyTorch for a Navier-Stokes Inverse Problem

### Recovering blood flow and an unknown viscosity in a stenosed artery from 20 noisy sensors, with no PINN library

![Twenty sensor locations over a PINN-reconstructed blood-flow field in a narrowed artery, with the wall coloured by inferred wall shear stress](figures/phase1_hero.png)
*The PINN's velocity field in a stenosed channel. Red rings are the twenty sensors, the only measurements it saw. The wall glows where the PINN infers the most shear stress. This is model output, not ground truth.*

I gave a physics-informed neural network (PINN) twenty noisy velocity readings from a narrowed artery and asked for two things nobody measured: the whole flow field and the viscosity of the fluid. The field came back within 4.7%. The viscosity came back 21.8% high. Wall shear stress, the quantity the exercise was for, looked like it matched the CFD reference, and it took me a while to see why that was wrong.

This article builds that inverse Navier-Stokes problem in PyTorch from scratch, without DeepXDE, and reports what worked and what fooled me.

> What you'll get: a Navier-Stokes residual in about fifty lines of plain PyTorch, a learnable viscosity that was 9.1% off at epoch 8,200 and 21.8% off at epoch 15,000 with nothing in the loss curves to say which one to trust, and a wall shear stress error of 24.4% that hid two larger errors pointing in opposite directions.

Wall shear stress is the frictional drag of blood along a vessel wall, and it is close to unmeasurable in a clinic. Doppler ultrasound loses accuracy near a boundary, and 4D-flow MRI has to differentiate a velocity field sampled at millimetre-scale voxels. The consensus statement notes that resolution adequate for flow volume is not necessarily adequate for quantities involving velocity derivatives [1], and reported values come in low against CFD by factors of three to six.

So it gets inferred, usually from a patient-specific CFD model. I wanted to try the inverse route: reconstruct the field from sparse noisy point measurements, the way a diagnostic workflow would have them. Twenty velocity probes with 7% noise, no pressure data, and viscosity unknown and recovered during training. Part 2 takes the same problem to a pulsatile cardiac cycle.

## Writing the Navier-Stokes residual in fifty lines of PyTorch

A PINN is an MLP that maps coordinates to physical quantities. Here, `(x, y) -> (u, v, p)`: two velocity components and pressure.

The loss is where the physics enters. Alongside a supervised term fitting whatever measurements exist, you add a term built from the governing PDE's *residual*: how badly the network's own output violates the equation, obtained by differentiating the network with respect to its inputs. For incompressible Navier-Stokes that means continuity plus one momentum equation per velocity component.

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

Drive those three toward zero at points scattered across the domain and the network stops interpolating your twenty data points. It becomes closer to a solution of the PDE that also agrees with your measurements.

The flag that matters is `create_graph=True`, which lets you differentiate a derivative. The viscous term needs `u_xx`, and without the flag the second call raises on the first epoch.

## Making viscosity a learnable parameter in an inverse PINN

In the forward problem the physics is fully specified and you want the field. A conventional solver does that job, usually faster than a PINN. The inverse case gives you scattered noisy measurements and something unknown about the system, and you want the field *and* the missing piece. Garay et al. infer reduced-order model parameters alongside velocity fields from noisy aortic data [2]. Arzani et al. recover near-wall blood flow from sparse measurements [3], and Satyadharma et al. look at how PINN performance holds up with sparse noisy velocity data [4].

My unknown is viscosity. It isn't a constant passed into `ns_residuals`. It's a parameter optimized jointly with the network weights, held in log space so it stays positive whatever the optimizer does:

```python
log_nu = torch.nn.Parameter(torch.tensor(float(np.log(4.0 * nu_true_hint))))
```

Log space matters here. The nondimensional viscosity is `0.005`, two and a half orders of magnitude below the scale the network weights live at, and Adam's step sizes are tuned for the latter. Optimizing `log_nu` puts the parameter on a scale where a step means something, and makes negative viscosity unrepresentable.

The `4.0` starts it at four times the true value, since a parameter that starts on the answer is a very expensive confirmation of something you already knew. A factor of four is still an anchor. The estimate approaches the true value from that side and never crosses it, and a literature value for blood would be a more defensible starting point than a multiple of the answer.

## Stenosed artery setup: 2D Navier-Stokes at Re = 200

The domain is a 2D channel of height `H` and length `5H`, narrowed in the middle by a one-sided Gaussian bump. I parameterized the narrowing by percent diameter stenosis, which is what the clinical literature specifies, and used 50%. One-sided on purpose: plaque is rarely concentric, and a symmetric bump can be fit by a mirror-image average. I only evaluate the stenosed wall below.

The physics is incompressible Navier-Stokes at `Re = 200`, nondimensionalized by channel height and inlet velocity. These are the same three equations `ns_residuals` drives to zero:

```
continuity:  du/dx + dv/dy = 0
momentum-x:  u*du/dx + v*du/dy + dp/dx = nu*(d²u/dx² + d²u/dy²)
momentum-y:  u*dv/dx + v*dv/dy + dp/dy = nu*(d²v/dx² + d²v/dy²)
```

The `nu` here is *kinematic* viscosity, since there is no density factor. Blood's dynamic viscosity is about `3.5 × 10⁻³ Pa·s` and its density about `1060 kg/m³`, so `nu` is roughly `3.3 × 10⁻⁶ m²/s`, a bit over three times water's. `Re = 200` is a mean velocity around `0.11 m/s` in a 6 mm carotid, the slow end of physiological. A higher Re would be more realistic and harder to converge, and I chose getting the pipeline working.

Ground truth comes from a Chorin projection solver I wrote from scratch: staggered MAC grid, immersed-boundary masking for the stenosis wall, 640 by 128 cells. Getting it trustworthy took two bugs and one resolution limit that turned out not to be a bug, which is Part 2's opening act. Validation: Poiseuille profile error of `5.9e-5`, mass conservation within `0.1%` across stations, divergence at machine precision after projection. The PINN never sees any of it.

## Twenty noisy sensors and no pressure data

Twenty points, each carrying a velocity vector plus independent Gaussian noise. No pressure readings: pressure is inferred, like viscosity.

The noise standard deviation is 7% of the *mean inlet* velocity, not 7% of the local speed. Because the flow accelerates through the throat, the relative noise at these twenty probes has a median of 4.6%, from 2.8% in the fastest flow to 56.6% at one probe that landed in nearly stagnant fluid. So "7% noise" flatters the difficulty in the fast regions and badly understates it in the slow ones.

The sampler excludes a `0.08H` margin near every wall, so the network has to reach the near-wall region through the physics residual. That is where wall shear stress lives. (The margin is measured vertically, so on the steep flank the true clearance is smaller.)

I did not put probes where the interesting flow is. They are drawn uniformly from the fluid interior, and exactly one of the twenty landed within `0.5H` of the throat. I told myself that made it a fair test. Part 2 finds that uniform placement is the bottleneck for the pulsatile result and reserves probes for the stenosis, since a Doppler exam images the narrowing on purpose.

## PINN architecture and loss weights in PyTorch

```python
class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_layers=6, hidden_units=32):
        super().__init__()
        dims = [in_dim] + [hidden_units] * hidden_layers
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 3))  # (u, v, p)
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)  # Xavier-normal, zero bias

    def forward(self, *coords):
        inp = torch.cat(coords, dim=-1)
        out = self.net(inp)
        return out[..., 0:1], out[..., 1:2], out[..., 2:3]
```

It uses `tanh`, not ReLU. The momentum residual needs `u_xx`, and ReLU's second derivative is zero almost everywhere, so the viscous term's gradient signal disappears while your loss curve carries on looking healthy.

Six layers of 32 units is about 5,500 parameters. I started at 7x48 with 4,000 collocation points and gave up 55 minutes into a 15,000-epoch run. At 6x32 with 1,500 points a step costs 0.129 seconds on this CPU, about half an hour per run. I changed both at once, so I cannot tell you which mattered more.

The loss has three weighted groups:

| Group | Weight | What it contains |
|---|---|---|
| `data` | 100 | MSE against the twenty readings |
| `pde` | 1 | MSE of the three residuals at 1,500 collocation points |
| `bc` | 10 | no-slip on both walls (100 points each), the exact inlet profile (100 points), one point pinning `p = 0` |

Collocation points are drawn uniformly over the fluid by rejection sampling and resampled every 200 epochs, so the network can't settle into one fixed point cloud. The throat and the far field get the same density. Boundary points are sampled once and held fixed. The no-slip points follow the *smooth analytic* stenosis curve, not the solver's staircased mask, which is free accuracy a mesh-free method hands you. They are spaced uniformly in `x`, so the steep flank where shear stress peaks gets no extra resolution.

Incompressible pressure appears in the equations only as a gradient, so it is defined up to an additive constant. Without that one pinned point, nothing in the loss stops the network from settling on `p + 400`.

There is deliberately no outlet condition, on the reasoning that over-specifying an outflow is a good way to get a wrong answer confidently. Part 2 has a long story about what the weighting did to a later run, so treat `data=100, pde=1, bc=10` as a starting point. Training is Adam for 15,000 epochs, then 300 iterations of L-BFGS with strong Wolfe line search.

## The viscosity was right at epoch 8,200 and wrong at the end

Jointly optimizing a physical parameter with network weights sounds like it should be free: one more tensor in the optimizer. It wasn't. Viscosity is only identifiable once the velocity field is roughly right, and a scalar converges on a different timescale than a field does.

The first run put everything on one learning rate. Viscosity passed within 7.6% of the truth around epoch 1,200 and kept going, finishing at 0.00186 against a true 0.005, a 63% error, while the data and boundary losses looked fine throughout. L-BFGS then improved the PDE loss and pushed viscosity further the wrong way.

A parameter that sails through the right answer and out the other side is moving too fast relative to whatever constrains it. So I gave `log_nu` its own learning rate at 0.1× the network's, and froze it during L-BFGS, so the polish phase would refine the field and leave the constant alone. That run came in at 124% error. It approached the truth monotonically and did not arrive before training stopped. The failure mode had flipped, which put the answer somewhere between the two.

At 0.3×, with epochs raised to 15,000, it came down from above and got close without ever crossing: the best estimate was 0.00546 at epoch 8,200, a 9.1% error. Then it drifted back up and settled at **0.00609 against a true 0.005, a 21.8% error**.

```python
optimizer = torch.optim.Adam([
    {"params": model.parameters(), "lr": cfg["adam_lr"]},
    {"params": [log_nu], "lr": cfg["adam_lr"] * 0.3},
])
```

![Per-term training losses and the viscosity trajectory normalized by the true value](figures/phase1_training_curves.png)
*Top: unweighted per-term losses. Bottom: recovered viscosity divided by the true value. It approaches from above, is closest near epoch 8,200 at 9.1%, and drifts back up to 21.8%. The loss curves give no sign of the drift.*

The parameter was twice as accurate two thirds of the way through training as it was at the end, and nothing in the loss curves marks epoch 8,200 as special. If you recover a physical parameter this way, the final-epoch value is not obviously the one to report, and I have no principled stopping criterion to offer. Checkpoint the parameter trajectory and look at it.

The run was also budget limited. The PDE term is still descending in a straight line when Adam stops, having fallen another 31% over the last quarter of training, and it ends about 140 times above the data term.

## PINN results: 4.7% field error and a wall shear stress number I almost trusted

The field reconstruction comes in at **4.7% relative L2 error** on both `u` and velocity magnitude.

![Ground truth versus PINN velocity magnitude with sensors on both panels and the zero-u contour in red](figures/phase1_field_comparison.png)
*Ground-truth CFD (top) against the PINN (bottom), with the twenty sensors drawn on both. The red line is the `u = 0` contour: flow runs backwards below it. The PINN reproduces the reversed-flow region along the lower wall, plus a short extra fragment on the downstream flank of the bump.*

The `v` component alone sits at 29%. Channel flow is overwhelmingly axial, so `v` is small everywhere and the same absolute error divides by a much smaller number.

![Absolute and relative pointwise error maps](figures/phase1_error_heatmap.png)
*Top: absolute velocity error, peaking at the outlet corner and the throat crest, with a faint band along the inlet centerline. Bottom: the same error relative to local speed (floored at 0.05). The reversed-flow boundary and the flank of the bump light up here. Absolute error flatters them, because velocity is small there.*

Pressure was promised as an inferred quantity. No sensor measures it, and only its gradient appears in the equations, so the network is free to be off by any constant. Comparing the two fields after removing each one's mean gives **12.8% relative L2 error**. The pressure *drop* from end to end comes out at 3.30 against a true 3.16, about 4% high. That is the same kind of quantity fractional flow reserve is built on, and it falls out without ever being observed.

Wall shear stress, the quantity no sensor touched, comes back at **24.4% relative L2 error**.

![Wall shear stress along the stenosed wall, PINN against ground truth](figures/phase1_wss_comparison.png)
*Wall shear stress along the lower (stenosed) wall in nondimensional units. The ground truth is a raw finite difference off a staircased mesh, so it is noisy near the peak. The PINN is a smooth continuous function.*

Those two curves look closer than 24.4% sounds, and when I went to explain why, I found I had been fooling myself. Wall shear stress here is `nu` times a velocity gradient, and *both* factors are inferred. My evaluation compared the PINN's recovered viscosity times the PINN's gradient against the true viscosity times the true gradient. Separate them and the picture changes. The recovered viscosity is 21.8% high. The network's near-wall gradient is 22% low across the whole wall, and at the station where the true gradient peaks it reads 0.766 of the truth. The two errors point in opposite directions and largely cancel in the product, which is why the peak looked well matched at 0.316 against 0.335. Feed the true viscosity into the same network and the peak drops to 0.259.

![Three panels: wall velocity gradient, viscosity, and wall shear stress with the true-viscosity curve](figures/phase1_cancelling_errors.png)
*Gradient times viscosity equals wall shear stress. The PINN's gradient is low (1), its viscosity is high (2), and the product looks fine (3, dashed). The dotted curve is the same network with the true viscosity, which is what the gradient alone gives you.*

So there is a systematic underestimate, and it was hiding inside a quantity that looked fine. Damping of the near-wall gradient is the failure Part 2 runs into much harder. If you take one thing from this article, take this: a product of two inferred quantities can agree with the truth while neither factor does. Report the factors.

The 24.4% was 97.8% before I found a bug, and the bug was in my reference calculation. I had been dividing the near-wall velocity by the distance from a grid cell centre to the *continuous* analytic wall curve. At a few columns those nearly coincided, so I was dividing by almost nothing. Individual points looked reasonable. The array statistics were garbage. Estimating the gradient from two interior fluid points and the fixed spacing between them fixed it.

My divergence was at machine precision and my velocity error was under 5%, and none of that told me the *reference* was wrong by 50x at a handful of columns. The solution was fine. The thing measuring it was not.

Two caveats on the comparison. The PINN is evaluated exactly on the analytic wall, while the reference is a two-cell difference slightly above the staircased wall. And both compute `nu du/dy`, which is the true wall shear only where the wall is horizontal: exact at the crest, an approximation on the inclined flanks, where most of the disagreement sits.

<!-- ABLATION: zero-sensor result goes here once phase1_nosensor.pt is evaluated -->

## Five things to check before you build your own inverse PINN

Give any learnable physical parameter its own learning rate, and expect to tune it separately from the network's. The coupling between parameter and field is the whole difficulty, and it doesn't announce itself in the loss curves. All three of my viscosity runs had healthy-looking data and boundary losses.

Watch the per-term losses. A total dominated by a `data` term weighted at 100 can fall steadily while the physics term climbs.

Write a manufactured-solution test for the residual code. It costs about twenty lines: pick something polynomial, derive the residuals by hand, assert they match. Mine caught an autograd edge case before it reached a training run.

Measure your compute budget before committing to an architecture. Every runtime estimate I made in this project needed checking.

And when the quantity you care about is a product of two inferred things, report the factors.

## Where this breaks

The domain is too short. Behind the bump the flow separates and runs backwards along the wall, and in a 5H channel that reversed region never closes. It is still going at the outlet, where the first few fluid cells read `-0.004` to `-0.054`. An outflow condition assumes fluid is leaving, so wherever it comes back in, the boundary is telling the reference solution what to do. That makes the ground truth downstream of about `x = 2.2` less trustworthy than it looks. The shear stress peak at `x ≈ 1.9` sits upstream of the reversal. A longer channel is the fix, and it means regenerating everything.

Blood is also not Newtonian. It is shear-thinning, so its viscosity climbs as the flow slows. A constant viscosity is standard for large arteries, where shear is moderate to high, and it is what makes "recover a single viscosity" a coherent question. The assumption is weakest in the separated zone behind the stenosis, the same place the domain is too short.

The network also gets exact boundary data: the inlet profile, the no-slip walls, and the geometry. "From twenty sensors" describes the interior measurements only.

The rest is scope. The flow is 2D, the walls are rigid, and the noise is additive isotropic Gaussian, which stands in for Doppler without simulating beam angle, speckle, or finite sample volume. The link between wall shear stress and plaque location is well studied but contested [5].

The steady case gave me a field I could trust to 5% and a wall shear stress I could not, until I split it into its two factors. Part 2 turns the flow pulsatile, and the same damping comes back much harder: the PINN's peak reaches 27% of the true peak. It took three fixes and two failed experiments to get anywhere. If that is the story you want, follow along.

If you have recovered a physical parameter with a PINN, I'd like to know whether your best epoch was your last.

All code, configs, and figures: [repo link].

---

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
