<!--
SUPERSEDED FOR PART 1. The live Part 1 draft is article/part1_draft.md.
This file is kept as raw source material for Part 2: sections 9 (cardiac
cycle), 10 (Phase 2 results) and 11 (where Claude helped and where it did
not) have not been rewritten anywhere else yet. Its title and section
numbering predate writing-guidelines.md and do not follow it; do not
lift prose from here without running it through that checklist first.

Draft for the Towards Data Science article. SEO target: "physics-informed
neural network" + "blood flow"/"stenosis" + "Python tutorial" +
AI-assisted-development framing, a long-tail combo with essentially no
readable competing content today.

Status: revised after a fresh-eyes editorial critique (see dev-log for
the fact-check fix log). Citations verified against real papers, see
References. Checklist being applied: writing-guidelines.md.
-->

# Seeing the Invisible

### Reconstructing blood flow and wall shear stress from sparse sensors with a physics-informed neural network in Python

## 1. The measurement blood flow can't give you

Somewhere along an artery narrowed by plaque, blood is dragging across the
vessel wall harder than almost anywhere else in the body, and nobody can
measure it. Not with an ultrasound probe, not with an MRI, not with any
clinical device in routine use. What clinicians *can* measure is a handful
of point velocities: a Doppler ultrasound beam gives you speed at wherever
you happen to aim it, plus a healthy dose of noise. Everything else about
the flow field, in particular the wall shear stress that actually
correlates with where plaque forms and ruptures, has to be inferred.

That's precisely the shape of problem a physics-informed neural network
(PINN) is built for: sparse, noisy point data, a physical law you trust
completely (here, the incompressible Navier-Stokes equations), and a gap
in between that the physics is asked to fill. This article builds one
from scratch in Python, for flow through a stenosed (narrowed) 2D vessel,
first steady, then pulsatile to match an actual cardiac cycle, and along
the way infers the one other thing that's never measured directly: the
blood's viscosity.

Two things set this apart from most PINN tutorials floating around. The
problem is a *reconstruction and inference* task on synthetic-but-honest
data, not a network re-deriving a known analytical solution. And the
whole thing was built with Claude as a coding collaborator, with an
unedited log of where that help actually mattered and where it didn't
(Section 11).

## 2. Why wall shear stress matters, and why nothing measures it directly

Wall shear stress (WSS) is the frictional force blood exerts on the
inside of a vessel, tangent to the wall. Vascular endothelial cells, the
layer lining every blood vessel, respond biologically to it: healthy,
steady, moderate WSS keeps that lining doing its job. Low or oscillating
WSS, exactly what happens in the recirculation zone just past a stenosis,
is associated with the endothelial dysfunction that precedes
atherosclerotic plaque formation and, in already-narrowed vessels, with
plaque instability and rupture risk [1].

The catch: WSS is a *near-wall velocity gradient*, literally `nu * du/dy`
evaluated as close to the wall as physically possible, and every
practical flow-measurement technique gets worse exactly where you'd want
to evaluate that gradient. Doppler ultrasound's spatial resolution and
beam-angle sensitivity degrade near a boundary. 4D-flow MRI voxels run
millimeters across, far too coarse for the boundary layer where the
gradient actually lives. In practice, WSS is almost never measured. It's
estimated from a coarser velocity field via post-hoc extrapolation, or
simulated outright in patient-specific CFD models built from imaging
geometry. This project takes the CFD-adjacent route but inverts it:
instead of solving the forward problem from a fully specified geometry
and boundary conditions, it reconstructs the field, WSS included, from
sparse, noisy point *measurements*, the way a real diagnostic workflow
would actually have them available.

## 3. Physics-informed neural networks and the inverse-problem angle

A physics-informed neural network is, mechanically, an ordinary
multilayer perceptron mapping spatial (and, here, temporal) coordinates
to physical quantities: `(x, y) -> (u, v, p)` for velocity and pressure.
What makes it "physics-informed" is the loss function. Alongside a normal
supervised term fitting available data, a second term is built from the
governing PDE's *residual*, evaluated via automatic differentiation
through the network itself:

```python
def _grad(outputs, inputs):
    # allow_unused/materialize_grads: needed once a quantity is
    # analytically independent of one input (e.g. d/dy of something
    # that only depends on x), PyTorch otherwise raises instead of
    # returning zero.
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

Push these three residuals toward zero at points scattered through the
domain, and the network is pushed toward a solution of the PDE, not just
an interpolant of the sensor data. That's what lets a network trained on
a sparse handful of points still produce a physically consistent field
everywhere else.

This project's use of the idea is less common than the typical PINN demo
in two ways. First, it's an **inverse** problem: viscosity
(equivalently, the Reynolds number) isn't a known input baked into the
residual above; it's a free parameter, jointly optimized alongside the
network's weights, the same way the boundary and data terms are. Second,
there's nothing analytical to check the answer against. Ground truth
comes from an independent numerical solver (Section 5), and the PINN
never sees anything from it except a sparse, corrupted sample.

PINNs for stenosed-vessel and cardiovascular flow are a real, active
research area. Garay et al. jointly infer reduced-order model parameters
and full velocity fields from scattered, noisy aortic measurements [2].
Ueda et al. reconstruct pulsatile post-stenotic flow fields from
transient velocity snapshots without ever feeding the network time as an
explicit input [3]. Separate work covers near-wall reconstruction
specifically [4], and how inference accuracy for a parameter like
viscosity degrades as sensor count drops [5]. What's comparatively rare
is an accessible, runnable, start-to-finish build of one of these: most
of what's public is a paper's method section or a repository with no
walkthrough. That gap is this article's reason for existing more than any
claim of novel science.

One more piece of context. 2025 saw a wave of work using LLMs as
collaborators in building PINNs themselves, not just as flow solvers.
Lang-PINN coordinates four cooperating agents (parsing a task description
into a PDE, choosing an architecture, generating code, diagnosing
failures) to go from a natural-language problem statement to a working
network [6]; PINNsAgent automates architecture and hyperparameter choices
for PDE surrogates the same way [7]. This project doesn't build a
framework, and the governing PDE was never in question. What an assistant
like Claude actually does or doesn't help with, on a single concrete
project, is a narrower and more useful question, logged as it happened
rather than reconstructed afterward (Section 11).

## 4. Setting up the problem

**Geometry.** A 2D channel of height `H`, length `Lx = 5H`, with an
eccentric (one-sided) Gaussian-bump stenosis: `h(x) = A * exp(-(x-x0)^2 /
(2*sigma^2))`, parameterized by clinical **percent diameter stenosis**
(%DS) rather than a raw bump height, since that's the number this problem
is actually specified in. Eccentric rather than symmetric on purpose: a
one-sided narrowing produces asymmetric wall shear stress on the two
walls, both more physiologically realistic and a better test of whether
sparse-sensor reconstruction can resolve near-wall asymmetry rather than
a mirror-symmetric average.

**Governing equations.** Incompressible Navier-Stokes, nondimensionalized
by channel height `H` and inlet mean velocity `U0`:

```
continuity:  du/dx + dv/dy = 0
momentum-x:  u*du/dx + v*du/dy + dp/dx = nu*(d2u/dx2 + d2u/dy2)
momentum-y:  u*dv/dx + v*dv/dy + dp/dy = nu*(d2v/dx2 + d2v/dy2)
```

Phase 2 adds `du/dt` and `dv/dt` local-acceleration terms to the two
momentum equations (Section 9).

**Reynolds number.** `Re = U0*H/nu = 200` for the steady phase, confirmed
empirically to produce a real, stable post-stenotic recirculation zone
without shedding at the resolution this project simulates at, rather than
assumed from the parameter alone.

**Boundary conditions.** Inlet: a prescribed, exact parabolic (Poiseuille)
profile, chosen to remove any entrance-length ambiguity from what the
ground-truth solver has to converge to. Outlet: zero velocity-gradient
plus a Dirichlet `p=0` pressure reference, needed to remove the pressure
null space an otherwise all-Neumann Poisson problem would leave
undetermined. Walls and the stenosis surface: no-slip.

**Nondimensionalization matters more than it looks.** Real blood
viscosity is on the order of `1e-3 Pa*s`, wildly out of scale with the
`O(1)` network weights a PINN's optimizer is tuned for. Working in
`H`/`U0` units throughout, and inferring nondimensional viscosity (only
converting back to a dimensional number for final reporting), sidesteps a
whole category of optimizer-scaling headaches before they start.

## 5. Building synthetic reality: a Navier-Stokes solver from scratch

There's no ground truth to reconstruct against unless something produces
one first, and unlike a typical inverse-problem demo that inverts a known
analytical field, this project needed a genuine numerical flow solver.
Rather than reach for a full finite-element package (no GPU or conda on
the build machine, and a black-box FEM call is a lot less explainable in
an article than a solver built by hand), the ground truth comes from a
from-scratch **Chorin projection method** (fractional-step) solver on a
**staggered MAC grid**, with the stenosis represented by **immersed-
boundary cell masking**.

The mechanics, briefly: each timestep advances velocity explicitly under
convection and diffusion (advection discretized with a donor-cell/upwind
blend, needed for stability once the flow accelerates through the
narrowed throat), solves a pressure Poisson equation whose right-hand
side is exactly the resulting divergence, then subtracts the pressure
gradient to project back onto a divergence-free field. The stenosis wall
is a static fluid/solid cell mask: no-slip is enforced by zeroing
velocity on solid faces, and the Poisson matrix gets identity rows at
solid cells and no-flux treatment at the fluid cells beside them.

This is also where the build's real debugging happened: three distinct
issues, only one of which was a genuine code bug in the usual sense (a
second, subtler bug lived in the outlet boundary handling, and the third
was a real numerical-resolution limit rather than a bug at all). The full
account of finding and fixing all three is in Section 11. The validated
result, once fixed: Poiseuille-profile error of `5.9e-5`, mass
conservation within `0.1%` across stations, and machine-precision
divergence (`~2e-13`) after each projection step.

## 6. Simulating imperfect measurements

The PINN never sees the ground-truth field directly. It sees a small,
fixed set of point locations (20 for the steady case), each sampled once
(or, in the pulsatile case, repeatedly through the cycle, Section 9),
with independent additive Gaussian noise on the velocity reading. This
stands in for sparse Doppler-ultrasound-style point velocity probes, at
roughly the level of physical detail used in the cited inverse-blood-flow
literature. It is *not* a simulation of actual Doppler physics: beam
angle, speckle, and a finite sample volume are all real effects this
doesn't model, stated here plainly rather than glossed over.

Two of those choices are easy to get quietly wrong in a way that flatters
the result, so they're worth stating plainly. Sensor locations exclude a thin
margin near every wall, which makes the problem harder on purpose: that
margin is exactly the near-wall region the article's headline result
needs, so the PINN never sees it directly and has to infer it from the
physics residual alone. And no pressure sensors exist at all. Pressure
and viscosity are purely inferred quantities here, never fitted directly.

## 7. Building the physics-informed neural network in PyTorch

**Architecture.** A plain multilayer perceptron, `tanh` activations
throughout, not ReLU (the viscous term needs a clean *second* derivative
through the network, and ReLU's second derivative is zero almost
everywhere), Xavier-initialized:

```python
class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_layers=7, hidden_units=48):
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

`(x, y)` maps to `(u, v, p)` in Phase 1; Phase 2 adds `t` as a third
input. The `7x48` shown above is the class default; every run in this
article actually used `6x32`, downsized from the original plan after
timing it directly showed 7x48 took roughly 55 minutes per training run
on this CPU, versus 6x32's 7-10 minutes for comparable convergence.
Viscosity is **not** a network output. It's a separate, directly
optimized parameter held in log-space (`nu = exp(log_nu)`), so it stays
positive by construction no matter how the optimizer moves it, and
deliberately initialized 4x away from the true value so recovering it is
a real test rather than a warm start in disguise.

**Loss terms.** A data-fit MSE at the sensor locations. The PDE-residual
MSE from Section 3, evaluated at points resampled periodically through
training so the network can't overfit to one fixed point cloud. A
boundary-condition term: no-slip sampled along the *smooth analytic*
stenosis curve, notably not the solver's staircase mesh, an advantage of
a mesh-free method here, plus the exact inlet profile. And a
pressure-gauge term pinning `p=0` at one point, since incompressible
pressure is only ever defined up to an additive constant, and nothing
else in the loss would stop the network from drifting to `p + const`.
These combine with fixed weights (`data=100, pde=1, bc=10`), a starting
point from common practice in the cited literature rather than a resolved
optimum. Section 11 covers a case where that particular weighting choice
mattered a lot more than expected.

**Training.** Two stages: Adam for the bulk of training, then L-BFGS for
a shorter polishing phase once the loss landscape is calm enough for a
quasi-Newton method to behave well. The learnable viscosity parameter
gets its own separate learning rate and is frozen entirely during the
L-BFGS phase, for reasons that took real trial and error to land on
(Section 11).

---

## 8. Phase 1 results

Trained on 20 sensors and nothing else, the network recovers the velocity
field to a relative L2 error of 4.7% (measured on both `u` and
`|velocity|`; `v` alone runs to 29%, which sounds alarming until you
remember `v` is the small transverse component here: a channel flow's
velocity is overwhelmingly axial, so the same absolute error reads as a
much larger percentage of a much smaller quantity). Side by side with the
ground truth, the reconstructed field reproduces the acceleration through
the throat and the recirculation zone immediately downstream of it, the
two features that actually matter physiologically, not a blurred average
of the sensor readings (`article/figures/phase1_field_comparison.png`,
`phase1_error_heatmap.png`).

Viscosity, the inverse-problem target that started deliberately 4x off
from the truth, settles at 0.00609 against a true value of 0.005: 21.8%
error. Three learning-rate schedules for the learnable parameter were
tried before one converged inside a reasonable epoch budget instead of
overshooting or crawling (the full trajectory is in Section 11). 21.8% is
the final number, reported as-is.

The headline result is wall shear stress: something no sensor in this
setup measures directly, reconstructed at a relative L2 error of 24.4%
(`phase1_wss_comparison.png`). The PINN's WSS curve is visibly smoother
than the ground truth's, a continuous function from a neural network
against a finite-difference estimate off a staircase-boundary mesh, while
still tracking the true curve's peak at the throat and its sign change
through the recirculation zone. This number started at a meaningless
97.8% before a post-processing bug was found and fixed (Section 11); 24.4%
lands in the same range reported for comparably sparse setups in the
cited literature [2, 5].

## 9. Raising the stakes: simulating a cardiac cycle

A steady flow is a reasonable first target, but it's not what blood
actually does. Phase 2 replaces the constant inlet with a waveform
standing in for a cardiac cycle. Reconstructing how wall shear stress
rises and falls through that cycle, not just what it looks like on
average, is arguably the more clinically relevant question anyway: an
artery spends most of a heartbeat somewhere other than its time-averaged
state.

**The waveform.** A plain `1 + A*sin(2*pi*t/T)` is symmetric about its
peak; it reads as sloshing, not beating. The inlet here instead follows a
two-harmonic sum, `mean + A1*sin(phase) - A2*cos(2*phase)`, the minimum
Fourier truncation that can skew the peak into the first quarter of the
cycle (systole) and stretch the return to baseline (diastole) while
staying perfectly smooth: the PINN needs an autograd derivative through
this function, so a kink anywhere would show up directly in the
acceleration term. The two coefficients (0.55, 0.20) were picked by
scanning numerically for two checked, not assumed, properties: the
waveform stays strictly positive everywhere (no momentary inlet backflow
to complicate the boundary condition), and its peak sits inside the first
quarter of the cycle.

**Reaching a real limit cycle.** The ground-truth solver has to run
through several cardiac cycles before the flow settles into a repeating
pattern, and "several" turned out to mean more than a first guess
suggested. The channel's domain-transit time (length divided by inlet
velocity) is close to five cardiac periods, so a disturbance from the
startup transient takes that long just to clear the domain once.
Measuring the cycle-to-cycle periodicity residual directly showed 8.6%
remaining after 6 cycles and 1.5% after 10. Since the whole 10-cycle run
costs only a few minutes at this resolution, there was no reason to
settle for the noisier number.

**Teaching the network to be periodic.** The PINN only ever sees one
representative cycle of sparse data, not the multi-cycle transient the
solver ran to get there. A soft loss term telling it directly "assume the
flow at time `t` matches the flow at `t + T`" does the rest. A cleaner
version of the same idea shows up again in Section 11: replacing the raw
time input with sine/cosine features at the cardiac harmonics makes the
network exactly periodic by construction instead of just softly
encouraged toward it.

**Sensors.** Twenty probes again, each now read at 18 points through the
cycle instead of once, standing in for a fixed Doppler probe recording
continuously rather than a fresh random location every instant.
Placement is deliberately not fully uniform this time: 30% of the probes
are reserved for the stenosis neighborhood specifically. That's both the
more clinically honest choice (a real Doppler exam images the suspected
narrowing on purpose) and, it turned out, load-bearing for the final
result (Section 11).

**Warm start.** The Phase 2 network initializes from Phase 1's converged
weights rather than from scratch. Phase 1's first layer maps `(x, y)` to
hidden units, Phase 2's maps `(x, y, t)`, one extra input column, so every
trained weight except that new column transfers directly. Viscosity isn't
re-inferred in this phase; it's carried over from Phase 1's validated
estimate, since Phase 2's measurements are spread thinner across time
than Phase 1's were concentrated in space, and re-fitting a single scalar
parameter benefits from more data, not less.

## 10. Phase 2 results

Over the full recorded cycle, the reconstructed velocity field holds a
mean relative L2 error of 15.1%, peaking at 19.5% at the hardest phase of
the cycle, worse than Phase 1's steady 4.7%. That's the expected cost of
asking the same 20 probes to now also resolve *when* things happen, not
just where.

Wall shear stress at the throat is the number that took the most work to
get right, and it's still the weakest one in the project: 70.6% relative
L2 error, down from a first attempt that was, plainly, unusable (a
near-flat prediction against a ground truth that actually swings from
0.04 to 0.53 through the cycle, `article/figures/phase2_wss_time_trace.
png`). What changed between "unusable" and "70.6%" is the subject of the
next section, and it matters for reading this number correctly: the
final reconstruction gets the *timing* right, peaking at essentially the
correct point in the cycle, and even reproduces a secondary bump in the
diastolic recirculation zone that has no reason to appear by chance. What
it still gets wrong is *amplitude*: the PINN's peak reaches only 0.145,
about 27% of the true 0.527. That's a real, specific, named limitation
(Section 11 explains why), reported at its actual size rather than
rounded toward something more flattering.

The field-comparison animation (`phase2_field_animation.gif`) is the most
convincing single artifact in the whole project. Watching the
reconstructed and true velocity fields pulse through a cycle side by side
makes both the timing accuracy and the amplitude gap legible in a way no
single error metric fully captures.

## 11. Where Claude helped, and where it didn't

This project was built with Claude as a coding collaborator throughout.
The answer to "did that help" is: yes, unevenly, in ways specific enough
to be useful to someone deciding whether to try the same thing.

**Deriving physics from a stated argument, and turning it into code, is
where it earned its keep most directly.** Working through the
finite-volume Poisson stencil for the immersed-boundary solver face by
face, why a solid neighbor contributes nothing, why a Dirichlet-outlet
neighbor contributes `-2/h^2` to the diagonal only, is exactly the kind
of task an LLM handles well. Understanding *why* each face behaves that
way, rather than copying a formula, is also what made the bugs below
findable at all instead of a black box that just diverged. The same
strength showed up in bug isolation once given a concrete, reproducible
symptom to instrument: three separate numerical bugs surfaced during
solver validation (a spurious pressure gradient at solid/fluid
interfaces, an inconsistent outlet boundary condition, and a staircase-
resolution limit that turned out not to be a bug at all), all three found
by tracing values at specific grid locations across specific timesteps,
not by guessing at fixes and checking if they helped.

**It was not a substitute for actually running things.** Every budget and
convergence question in this project, does this network size fit the
CPU time available, has the solver reached steady state, has the
pulsatile flow reached periodicity, was answered by measuring, and
several of those measurements contradicted the first assumption. The
original Phase 2 config still carried a 320x64 grid that Phase 1's own
validation had already shown was too coarse for a 50%-stenosis case;
catching that meant re-reading the dev-log's bug history against the
current config, not any special insight. A 3-cycle periodicity assumption
turned out to need 10, found by measuring the residual rather than
trusting a round number.

**The clearest case study is the Phase 2 wall-shear-stress debugging
arc**, three fixes with two different failure modes along the way, a real
instance of diagnose-hypothesize-test-cheaply rather than a straight
line to the answer.

1. The first full Phase 2 training run converged cleanly on every loss
   term except one: the PDE residual actually *climbed* during training
   instead of settling. Phase 2's momentum equations carry an extra
   acceleration term (`u_t`, `v_t`) that Phase 1's steady residual never
   had to satisfy, and at Phase 1's inherited loss weight, it simply lost
   the optimization tug-of-war against the data and boundary terms. A
   cheap A/B (1,800 epochs, a few minutes each) comparing the original
   weight against a 20x larger one confirmed the fix before the full
   ~25-minute run was repeated.
2. Fixing that surfaced the real shape of the problem: the field
   reconstructed reasonably well, but WSS(t) stayed almost flat. Checking
   a bulk point away from the wall directly, rather than trusting the
   aggregate error number, showed the network *did* learn real pulsatile
   time-dependence with roughly the correct phase, but compressed the
   peak-to-trough amplitude substantially, and since WSS is a spatial
   derivative of an already-damped near-wall field, that damping shows up
   amplified in the derived quantity. The fix was architectural:
   ```python
   def forward(self, *coords):
       *space, t = coords
       time_feats = [fn(t * freq * 2 * math.pi)
                     for freq in self.fourier_time_freqs
                     for fn in (torch.sin, torch.cos)]
       inp = torch.cat([*space, *time_feats], dim=-1)
       return self.net(inp)
   ```
   replacing the raw time input with sine/cosine features at the cardiac
   fundamental and second harmonic, so the network doesn't have to
   discover periodicity from scratch through tanh layers alone. A
   verifiable side effect: because `sin`/`cos` at `2*pi*k/T` are
   themselves exactly periodic, the network's output became exactly
   periodic in time by construction, confirmed by a manufactured-input
   test and by the soft periodicity loss reading a clean, unchanging zero
   for the rest of training.
3. The improvement was real but partial, so the next hypothesis was
   collocation density: maybe the physics constraint didn't have enough
   sample points near the throat wall to pin down a sharp gradient there.
   Two variants of concentrating collocation points near walls both made
   field *and* WSS error worse, a genuine negative result, and one that
   redirected the investigation correctly. Checking the *sensor* data
   itself, not the training loop, showed only 1 of 20 probes fell
   anywhere near the throat, and not even near the wall. The network had
   almost no direct measurement telling it what happens at the one
   location the headline result is evaluated. Reserving 30% of the sensor
   budget for that neighborhood, overruling pure randomness with a
   clinically defensible design choice, produced the final result above.

None of those three fixes came from an assistant's intuition about what
"should" work. Each came from checking a specific, falsifiable prediction
against actual numbers, including a willingness to report the two
collocation attempts that didn't pan out. That's the Lang-PINN/PINNsAgent
question from Section 3, answered narrowly instead of in general: an
assistant sped up the mechanical parts of the loop, writing the
instrumentation, proposing candidate fixes, explaining what a given
number implies, without ever replacing the loop itself.

That extended past code, too. At one point, four consecutive background
training runs were killed by something outside either the code or the
model, for reasons never identified. The fix was infrastructural, a
resumable, chunked foreground training script
(`scripts/_train_phase2_chunk.py`), not a data-science insight. Sometimes
"the run keeps getting interrupted" is just the bottleneck, with no
clever answer waiting underneath it.

## 12. Limitations and future work

This project simplifies the real problem in several specific, nameable
ways:

- **Two dimensions only.** A real stenosis produces secondary, swirling
  flow structure a 2D cross-section can't represent at all. This
  project's recirculation zone is real, but it's a 2D idealization of a
  3D phenomenon.
- **Rigid walls.** Real vessel walls are compliant; there's no
  fluid-structure interaction here, which matters most at high systolic
  pressure, exactly the part of the cycle this project's WSS result is
  weakest at.
- **An idealized sensor model.** Additive isotropic Gaussian noise stands
  in for Doppler ultrasound; it says nothing about beam-angle sensitivity,
  speckle, or a finite sample volume, all of which shape what a real
  probe returns.
- **A single severity and Reynolds number explored in depth.** How
  reconstruction accuracy degrades with sensor count and noise level was
  planned as a follow-up sensitivity sweep and didn't happen in this pass
 , a scope cut made for time, not a result being withheld.
- **Newtonian, constant viscosity.** Real blood is shear-thinning at low
  shear rates, most relevantly right where this project's recirculation
  zone sits.

## 13. Conclusion

A physics-informed neural network can reconstruct a clinically relevant,
unmeasured quantity, wall shear stress, from a sparse, noisy handful of
point velocity readings, in both a steady and a pulsatile setting. That
was true going in, on paper; what this project adds is the specific,
occasionally unflattering record of what it actually took to get there.
Phase 1 landed a clean, validated result across the board. Phase 2 landed
a partial one: correct timing, still-compressed amplitude, reported at
its real size. If there's one thing worth taking from the debugging arc
in Section 11, it's that the fixes that worked all came from checking a
specific number against a specific prediction, never from swapping in a
plausible-sounding change and hoping. An AI assistant made that loop
faster. It didn't make it optional.

All code, configs, the full unedited dev-log, and every figure in this
article are in the repository: [repo link].

---

## References

[1] P.D. Weinberg, "Haemodynamic Wall Shear Stress, Endothelial
Permeability and Atherosclerosis, A Triad of Controversy" (2022),
*Frontiers in Bioengineering and Biotechnology*, 10:836680.

[2] J. Garay, M. Dunstan, S. Uribe, F. Sahli Costabal, "Physics-informed
neural networks for parameter estimation in blood flow models" (2024),
*Computers in Biology and Medicine*, 178:108706.

[3] K. Ueda, H. Wakimura, S. Ii, "Reconstruction of instantaneous flow
fields from transient velocity snapshots using physics-informed neural
networks (Applications to pulsatile blood flow behind a stenosis)"
(2026), *Journal of Biomechanical Science and Engineering*, 21(2).

[4] A. Arzani, J.-X. Wang, R.M. D'Souza, "Uncovering near-wall blood flow
from sparse data with physics-informed neural networks" (2021), *Physics
of Fluids*, 33:071905.

[5] A. Satyadharma, M.-J. Chern, H.-C. Kan, Harinaldi, J. Julian,
"Assessing physics-informed neural network performance with sparse noisy
velocity data" (2024), *Physics of Fluids*, 36(10):103619.

[6] X. He, L. You, H. Tian, B. Han, I. Tsang, Y.-S. Ong, "Lang-PINN: From
Language to Physics-Informed Neural Networks via a Multi-Agent Framework"
(2025), arXiv:2510.05158.

[7] Q. Wuwu, C. Gao, T. Chen, Y. Huang, Y. Zhang, J. Wang, J. Li, H. Zhou,
S. Zhang, "PINNsAgent: Automated PDE Surrogation with Large Language
Models" (2025), arXiv:2501.12053.
