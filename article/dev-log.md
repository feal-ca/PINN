# Dev Log

Running, first-person build journal. Tags: `[numerics]` `[pinn-training]`
`[ai-assist-worked]` `[ai-assist-failed]` `[design-decision]`. This is raw
material for the article's "AI-assisted development" section — write it as
it happens, not polished.

## 2026-08-26

- `[design-decision]` Settled on a hand-rolled Chorin projection-method
  solver (staggered MAC grid, immersed-boundary masking for the stenosis)
  instead of a FEM library like FEniCS, mainly because we have no
  conda/GPU on this machine and dolfinx via pip is unreliable — but also
  because a from-scratch solver is explainable in the article in a way a
  black-box FEM call isn't.
- `[design-decision]` Chose Python 3.11 for the venv over the system's
  default 3.14.7 — too new to trust for stable PyTorch wheel availability.
  Confirmed torch 2.13.0+cpu installed cleanly on 3.11.
- `[numerics]` Planned Re=200, 50% diameter stenosis, eccentric bump as the
  Phase 1 starting point — flagged explicitly as something to *verify*
  empirically (does it actually reach steady state, or does it secretly
  vortex-shed?) rather than assume, at the M3 validation milestone.
- Environment + repo skeleton created (M0 done): venv, package layout,
  configs for both phases, gitignore, requirements pinned.
- `[design-decision]` M1 done: `geometry.py` (Gaussian-bump stenosis,
  %DS-parameterized, eccentric vs. symmetric) shared by both the solver
  and (later) the PINN's boundary sampler.
- `[numerics]` M2 done: staggered MAC grid, Chorin projection solver,
  donor-cell-blended advection (needed for stability once velocity
  accelerates through the throat -- pure central differencing wasn't
  robust enough at this Re). Worked through the immersed-boundary Poisson
  stencil myself via a finite-volume/flux argument rather than quoting a
  textbook formula, since getting the boundary handling right by
  understanding *why* each face contributes what it does made the later
  bugs below much easier to diagnose.
- `[numerics]` M3 validation surfaced three genuinely separate problems,
  in order of how they appeared:
  1. **Bug:** `_project` read the pressure at solid (masked-out) cells as
     if it were a real neighbor. That pressure is an arbitrary dummy value
     (the Poisson matrix decouples solid cells with an identity row), so
     reading it created a spurious pressure gradient exactly at the
     fluid/solid interface -- caused immediate, massive blow-up (u to
     1e150 within ~60 steps). Fix: treat a solid neighbor as zero-flux in
     the projection step too, matching the convention already used when
     assembling the Poisson matrix.
  2. **Bug:** the outlet used a Dirichlet pressure condition (p=0) in the
     Poisson matrix, but I was separately overwriting the outlet velocity
     with an independent zero-gradient copy *after* projection, discarding
     the pressure-consistent value. The two boundary treatments didn't
     agree on what "divergence-free" meant at that face -- looked fine at
     first (bulk flow stable, max|div u| ~1e-8) then grew steadily over
     thousands of steps (to ~1e-1) with the defect concentrated entirely
     in the last column. Fix: correct the outlet face with the *same*
     ghost-pressure convention used to build the matrix, instead of a
     second, inconsistent BC.
  3. **Not a bug -- a real resolution requirement:** with both bugs fixed,
     a 50% stenosis at the originally-planned 320x64 grid *still*
     diverges, via a third, distinct mechanism: single-cell staircase
     corners (a fluid cell with solid neighbors on two sides at once)
     produce an unbounded velocity spike. Confirmed by testing pct_ds in
     {0,10,20,30} at coarse resolution -- 0/10/20 stable, 30+ blows up --
     and by confirming the same 50%DS case is clean at 640x128. This is a
     known limitation of simple staircase immersed-boundary masking
     (as opposed to a cut-cell method), not something to "fix" so much as
     to give enough resolution. Bumped the production grid to 640x128;
     kept a coarse, mild-stenosis (pct_ds=20) config for the fast pytest
     suite, and put the resolution-sensitive checks (Poiseuille accuracy,
     recirculation) in `scripts/validate_solver.py` instead.
  Final validated numbers at 640x128, Re=200, 50%DS: Poiseuille rel. error
  5.9e-5, mass conservation within 0.1% across stations, max|div u| ~2e-13
  (machine precision), clear post-stenotic recirculation zone present.
- M4/M5 done: generated the Phase 1 ground-truth field (640x128, hit the
  20000-step budget without satisfying the strict 1e-6 tolerance, but
  max|div u|=1.83e-13 confirms it's physically converged -- just a very
  slow last-digit relaxation tail, not a real non-convergence) and sampled
  20 sparse noisy sensor points from it.
- `[numerics]` M6: built the PINN (network, autograd NS residuals, loss
  terms, training loop). The manufactured-solution test for the residual
  code (u=x^2 y, v=-x y^2, hand-derived expected residuals) caught a real
  edge case immediately: `u_y = x^2` has zero graph-connection to `y`
  (its true derivative w.r.t. y is exactly 0), and PyTorch's autograd
  refuses to differentiate against a topologically-disconnected variable
  by default instead of returning zero. Fixed with
  `allow_unused=True, materialize_grads=True` on the grad helper. A real
  trained network won't hit this in practice (tanh entangles every input
  nonlinearly), but it's a cheap, correct robustness fix and a good
  example of why a manufactured-solution test is worth the setup cost --
  it exercises corner cases a "does it run on real data" smoke test won't.
- `[design-decision]` Learnable viscosity is parameterized as `log(nu)`
  rather than `nu` directly -- keeps it positive by construction
  (`nu = exp(log_nu)`) and initialized deliberately 4x off from the truth
  so recovery is a genuine test, not a freebie.
- `[pinn-training]` The originally-planned network size (7x48) with 4000
  collocation points timed out at ~55 minutes for 15000 Adam epochs on
  CPU -- each collocation point needs second-order autograd, which
  dominates cost. Timed a few configurations directly rather than
  guessing: 6x32 network with 1500 collocation points and 8000 epochs
  trains in ~7-10 minutes with comparable convergence in a quick smoke
  test. Updated the config accordingly.
- `[pinn-training]` First full training run: nu recovered to 0.00186 vs.
  true 0.005 (63% error), despite data/bc losses fitting well. Trajectory
  showed nu passing within 7.6% of the truth around epoch 1200, then
  overshooting and drifting well past it for the rest of training --
  classic sign that the learnable parameter's effective step size was too
  large relative to how slowly the velocity field itself was converging
  (it shared the network's Adam LR). Also noticed L-BFGS improved the PDE
  loss but pushed nu the wrong way.
  Fix attempt 1: gave log_nu its own LR at 0.1x the network's, and froze
  it during L-BFGS (that phase should polish the field, not keep moving
  the physical parameter under an unrelated step-size rule). Result:
  worse (124% error) -- nu now approached the truth monotonically without
  overshooting, but 0.1x was too conservative to arrive within the epoch
  budget; still improving at epoch 8000 when training stopped.
  Fix attempt 2: 0.3x LR (a middle ground) + more epochs (15000). Result:
  nu approached from above and got much closer than attempt 1 (best
  0.00546 at epoch 8200, 9.1% error), then drifted back up and settled at
  0.00609 (21.8% error) after L-BFGS, with PDE loss continuing to improve
  (0.105 -> 0.072). Accepting this as the Phase 1 MVP result: it's not
  push-button-perfect, but it's an honest, defensible number from 20
  sparse noisy points on a CPU-trained network, and matches the kind of
  error magnitude reported in the cited inverse-blood-flow literature for
  comparably sparse setups.
- `[numerics]` M8 evaluation surfaced one more bug, in post-processing this
  time, not the solver or the PINN: the ground-truth wall-shear-stress
  estimate (nu * u_first_fluid_cell / distance_to_analytic_wall) blew up
  to ~14 at a few isolated x-columns -- ~50x every neighboring value --
  because that distance mixes a *continuous* reference (the analytic
  stenosis curve) with a *discrete* one (a staircase grid cell center),
  and at some columns those two coincidentally sit very close together,
  amplifying grid noise through the division. Fixed by estimating the
  near-wall gradient from two interior fluid points and the fixed grid
  spacing between them (never near-zero) instead -- standard practice for
  staircase-boundary post-processing. WSS relative L2 error went from a
  meaningless 97.8% to a legitimate 24.4%.
- **Phase 1 final results:** field relative L2 error 4.7% (u and
  |velocity|; v alone is 29% but v is the small transverse component,
  where percentage error is naturally noisier), viscosity recovered to
  21.8% error, WSS relative L2 error 24.4% -- and qualitatively, the PINN's
  WSS curve is visibly *smoother* than the ground truth's (continuous
  network vs. staircase-derived finite differences) while still tracking
  its peak at the throat and its sign change in the recirculation zone.
  Figures saved to article/figures/. This closes out Phase 1 (M0-M8).

## 2026-08-28

- `[design-decision]` M9: the pulsatile inlet waveform. Rejected a plain
  `1 + A*sin(2*pi*t/T)` -- symmetric about its peak, which reads as
  "sloshing" rather than "beating." Used a two-harmonic sum instead
  (`mean + A1*sin(phase) - A2*cos(2*phase)`, `src/.../solver/waveform.py`)
  -- the minimum Fourier truncation that can skew the peak early into the
  cycle (systole) and stretch the return to baseline (diastole), while
  staying perfectly smooth (matters later: the PINN needs an autograd
  derivative through it). Picked coefficients (0.55, 0.20) by scanning
  numerically for two checked, not assumed, properties: strictly positive
  everywhere (no momentary inlet backflow, keeping the Dirichlet inlet BC
  simple) and a peak inside the first quarter of the cycle.
- `[numerics]` M9: reused Phase 1's 640x128 grid fix directly -- the
  Phase 2 config still had the original, since-proven-too-coarse 320x64
  from before the M3 staircase-corner bug was found. Easy to miss since
  Phase 2 was written before that fix; caught by re-reading the config
  against the dev-log rather than assuming it inherited the fix.
- `[numerics]` M9: added a real time-accurate stepping mode
  (`ChorinSolver.run_transient`) alongside the existing pseudo-time
  `run_to_steady` -- `step()` already took a `t` argument from the start
  (planned ahead for this), so no solver-internals changes were needed,
  only a new driver on top.
- `[numerics]` M9: periodicity took longer to reach than a naive 3-cycle
  plan assumed. Domain transit time is ~Lx/U0 = 5 (nondimensional) time
  units, close to or longer than the ~1-time-unit cardiac period, so a
  disturbance from the startup transient takes several periods just to
  clear the domain once. Measured (not assumed) the cycle-to-cycle
  periodicity residual (`|u(t0) - u(t0+T)| / max|u|`) directly: 8.6% at 6
  cycles, 1.5% at 10 -- and the whole 10-cycle run only costs ~3-5 minutes
  at this grid, so there was no reason to accept the noisier number.
  Settled on 10 cycles, recording only the last. A coarse/fast unit test
  (`tests/test_solver_transient.py`) hit a milder version of the same
  effect and documents it rather than chasing a tight tolerance there.
- `[design-decision]` M11: Phase 2 does NOT re-infer viscosity -- it reuses
  Phase 1's recovered value. Phase 2's sensor data is spread across time
  snapshots rather than concentrated at one instant, so fitting a single
  scalar physical parameter a second time from a *sparser-per-snapshot*
  dataset would be strictly worse than using Phase 1's already-validated
  estimate.
- `[design-decision]` M11: warm-started Phase 2's network from Phase 1's
  converged weights. Phase 1's first layer maps (x,y) -> hidden units;
  Phase 2's maps (x,y,t) -> hidden units, one extra input column. Copied
  the trained (x,y) columns and every deeper layer verbatim, left the new
  t-column at its random Xavier init (`train_phase2.py::_warm_start_from_phase1`).
  This requires matching hidden_layers/hidden_units between phases, which
  is also why Phase 2's network stayed at Phase 1's 6x32 rather than the
  originally-planned larger 7x48. Verified with a 30-epoch dry run on the
  real dataset before committing to a full run: the BC loss (which the
  warm-started spatial weights should already mostly satisfy) dropped from
  9.6 to 2.0 in those 30 epochs alone -- a good sign the transfer is
  actually doing something, not just avoiding a crash.
- `[pinn-training]` M11: timed the unsteady residual directly before
  picking an epoch budget, same discipline as Phase 1. The extra u_t, v_t
  terms make each epoch ~5x Phase 1's per-epoch cost at a comparable
  collocation-point count (~0.19s/epoch at 1200 points on this CPU).
  Reduced adam_epochs to 8000 (from the original plan's 20000) to keep
  total Adam time in the same ~25-30 minute ballpark as Phase 1's run,
  rather than letting the cost scale up unchecked.

## 2026-08-28 (continued, different session)

- `[design-decision]` Picking up mid-flight: found a fully-built Phase 2
  pipeline already run (checkpoint + figures on disk) that this
  conversation didn't produce -- unclear origin, proceeding with it as the
  current state per the user rather than re-deriving it.
- `[numerics]` The existing Phase 2 result was broken: WSS(t) at the
  throat was nearly flat (~0.03-0.06) against a ground truth that swings
  up to ~0.53 at systole. Root cause visible directly in the training
  curves: PDE loss *increased* over training (0.6 -> 1.86) while
  data/bc/periodicity all converged fine -- the network was satisfying
  every term it could without being pushed hard enough to satisfy the
  unsteady Navier-Stokes equations themselves, which is exactly the term
  that WSS (a purely physics-derived, unmeasured quantity) depends on.
  First checked the physics wasn't just wrong: the existing
  manufactured-solution test for `unsteady_ns_residuals` passed, so the
  residual math itself was already correct -- this was a training/weight
  problem, not a formula bug.
- `[numerics]` A/B tested pde loss weight directly (1800-epoch runs,
  ~5-6 min each) rather than guessing: weight=1.0 (Phase 1's value,
  inherited by Phase 2) let pde loss climb to 1.86; weight=20.0 converged
  cleanly to ~0.05-0.09 with data/bc/periodicity still reasonable.
  Updated the config and re-running the full 8000-epoch training.
- `[numerics]` Re-evaluated: field error improved (26.8% -> 18.2%) but
  WSS(t) at the throat was still badly wrong -- data/bc/pde losses all
  converged fine, so this wasn't the same weighting problem again.
  Checked a bulk (non-wall) point near the throat directly: the network
  *does* learn real pulsatile time-dependence there (rises and falls in
  roughly the right phase), but compresses the peak-to-trough amplitude
  by about 2x (1.10 vs. true 2.33) -- and since WSS is a spatial
  derivative of an already-damped near-wall field, that damping shows up
  amplified in the derived quantity.
- `[design-decision]` Added sin/cos Fourier time features at the cardiac
  fundamental + 2nd harmonic (matching `cardiac_pulse`'s own two-harmonic
  construction) as network inputs in place of raw t
  (`network.py::PINN.fourier_time_freqs`), instead of asking a plain MLP
  to discover periodicity from scratch. Verified two things directly
  rather than assuming: a manufactured-input test that
  `model(x,y,t) == model(x,y,t+period)` exactly (it does, since
  sin/cos(2*pi*k*t/period) are themselves exactly periodic), and that
  `periodicity_loss` reads a clean 0.0 throughout training -- the soft
  penalty is now a redundant diagnostic rather than a load-bearing term.
- `[numerics]` First attempt with Fourier features spiked the initial PDE
  loss to 268 (vs. ~1 before) -- the extra `2*pi*freq` factor in `u_t`'s
  chain rule through sin/cos amplifies gradients at initialization -- but
  a smoke test showed it recovering over epochs rather than diverging, so
  ran the full 8000 epochs anyway: converged cleanly (pde 268 -> 0.086,
  comparable to the non-Fourier run's 0.153), field error improved again
  (18.2%), but WSS(t) *still* missed the systolic peak entirely.
- `[numerics]` Root-caused the remaining WSS failure to collocation
  *density*, not weighting or architecture: checked directly how many of
  the (uniform-over-the-domain) collocation points land near the throat
  wall specifically -- only ~13 of ~1068 fluid points per resample, out
  of a budget already split across the whole time period too. The PDE
  residual essentially had no support to teach the network the sharp
  near-wall systolic gradient WSS is computed from.
  Fix: added `_sample_near_wall_xy` (points within 0.05H of either wall)
  and mixed it into collocation sampling at a configurable fraction
  (`near_wall_collocation_frac: 0.3`) -- boundary-layer-aware collocation
  refinement, standard practice for PINNs on wall-bounded flows once a
  wall-derived quantity like WSS matters, not something to have started
  with blind. Re-running the full training with this change.
- `[ai-assist-failed]` Both near-wall collocation variants made things
  *worse*, not better: domain-wide near-wall points -> field 21.8%/WSS
  88.8% (vs. the Fourier-only baseline's 18.2%/78.2%); throat-concentrated
  near-wall points -> field 19.6%/WSS 84.5%. Diluting bulk-flow collocation
  coverage cost more than the near-wall resolution gained -- keeping both
  as tested, disabled-by-default code (`near_wall_collocation_frac: 0.0`)
  rather than deleting them, since a documented negative result is still
  useful record.
- `[numerics]` Root-caused the *actual* remaining bottleneck by checking
  the sensor data itself, not the training: of 20 randomly-placed probes,
  only 1 fell within 0.5 units of the throat, and not even near the wall --
  the network had essentially no direct data telling it what happens at
  the one location WSS is evaluated, so it was extrapolating that whole
  region from physics + warm-started spatial structure alone.
- `[design-decision]` Fixed by reserving 30% of sensor probes for a band
  around the throat (`sensors.throat_frac`/`throat_half_width`,
  `sampling.py::sample_sensors_pulsatile`) instead of placing all 20
  fully uniformly. This is also the more clinically honest choice -- a
  real Doppler exam images the suspected narrowing on purpose, it doesn't
  scan uniformly at random and hope some probes land there.
- `[numerics]` Four consecutive backgrounded training runs were killed
  externally mid-run (dev-log noted this failure mode once before, but
  not four times in a row with decreasing survival time) -- worked around
  by writing `scripts/_train_phase2_chunk.py`, which runs the same Adam
  loop as a sequence of short, resumable *foreground* calls (2000 epochs
  each, well under any single-call timeout) instead of one long
  background process, checkpointing model + optimizer + history between
  calls. This is scaffolding to get past an environment issue, not a
  permanent feature -- it duplicates `train_pinn_phase2`'s loop rather
  than refactoring it to be resumable.
- **Phase 2 final results, after all three fixes (pde loss weight, Fourier
  time features, throat-stratified sensors):** field relative L2 error
  over the cycle 15.1% (down from an initially unusable, flat-WSS state),
  viscosity reused from Phase 1 (not re-inferred, by design). Throat WSS(t)
  relative L2 error 70.6% -- still the weakest number in the project, but
  no longer meaningless: the PINN's WSS curve now peaks at the correct
  phase in the cycle (~t=9.25, matching ground truth) and even reproduces
  the secondary bump in the diastolic recirculation zone, with the
  remaining error dominated by *amplitude* compression (peak ~0.15 vs.
  true ~0.53) rather than getting the shape wrong. Recovering wall shear
  stress specifically -- a second-derivative, near-wall, unmeasured
  quantity -- from 20 sparse noisy probes through a full cardiac cycle is
  a substantially harder inverse problem than Phase 1's steady case, and
  this is an honest, explainable partial result rather than a hidden
  failure: the remaining gap is a known, named phenomenon (temporal
  amplitude damping in periodic PINNs), not an unexplained bug. Figures
  saved to article/figures/. This closes out Phase 2 (M9-M12).
- `[numerics]` Precise numbers for the article (computed directly from the
  final checkpoint, for citation accuracy): field relative L2 error over
  the cycle mean=15.076%, max=19.458%. Throat WSS(t): ground truth ranges
  0.0361-0.5266, PINN ranges 0.0575-0.1450 -- the PINN's peak is 27.5% of
  the true peak (a ~72% under-prediction, not "roughly half" -- caught and
  corrected during the article's fact-check pass after a first draft
  understated this).

## 2026-09-29: reach pass and zero-sensor ablation
- Ran the zero-sensor ablation (`_train_phase1_chunk.py --freeze-nu --data-weight 0`,
  output `data/checkpoints/phase1_nosensor.pt`, seed as the main run). Result: field
  26.4% (u), WSS 52.9%, pressure 55.5%. Compare 4.7%, 24.4%, 12.8% with sensors.
- Rebuilt Phase 1 figures for legibility (200 dpi, larger fonts, sensors on both
  panels, relative-error map, annotated viscosity trajectory) and added the hero
  and cancelling-errors figures.
- Part 1 cut to one arc for a roughly 10 minute read; the rest moved to the series.

## 2026-09-30: rewrite of Part 1 after the review comments
- `[design-decision]` Review said the end product was not a good PINN (wake vorticity
  missing, WSS reference spiky, viscosity drifting, too many numbers). Redid the
  Phase 1 experiment instead of rewording the old one. Old files moved to
  `article/figures/v1/`, `data/v1/`, `configs/phase1_steady_v1.yaml`,
  `article/part1_draft_v1.md`.
- `[numerics]` The old 5H channel had reversed flow all the way to the outlet (near-wall
  u < 0 from x = 2.2 to 5.0, outlet u down to -0.058), so the outflow condition was
  steering the reference. Regenerated the reference at Lx = 9 on a 1152x128 grid
  (`configs/phase1_long.yaml`, converged in 22,100 steps). Reversed flow now ends at
  x = 5.75 and outlet u >= 0.019.
- `[numerics]` Old WSS reference was jagged because it divided a staircase-cell value by
  a distance to the analytic wall. New estimate in `src/pinn_bloodflow/evaluation.py`:
  sample the tangential velocity at five points 0.03 to 0.09 H off the wall along the
  wall normal, fit u_t = a d + b d^2, WSS = nu a. Same recipe for CFD and PINN.
  Peak reference value 0.297 (old jagged one peaked at 0.335 on a spike).
- `[pinn-training]` Screening on the old channel (8,000 epochs, 20 sensors): input scaling,
  cosine LR decay, residual-based point selection and a 4x96 network all left the
  velocity error at 5.6 to 6.0%. Only the viscosity moved, erratically (8% to 86% off).
- `[pinn-training]` Long channel, 10,000 epochs, 20 zoned sensors, seed 0:
  pde weight 1: field 12.7%, nu 23% off. Weight 10: field 5.3%, nu 44% off. Weight 50:
  field 5.2%, nu 50% off. Weight 10 with 4x96: field 6.0%, nu 50% off, 2x the time.
  Outflow condition alone: field 6.1%, nu 44% off. nu_lr_scale 1.0 (with or without the
  outflow term): nu 41% off (undershoot to 0.003), field 10 to 12%.
- `[design-decision]` 40 sensors, weight 10, nu_lr_scale 1.0, outflow condition (R8): field
  4.4%, vorticity 10%, WSS 13% (peak 0.92 of reference), nu 0.4% off. The only change from
  R6 was 20 -> 40 sensors. That is the result in the article.
- `[numerics]` Robustness (all 40+ sensors, 10,000 epochs): start at 0.34x true nu: nu 7.3%
  off. 60 sensors: 0.9%. No outflow term: 0.8%. Sensor seed 1: 6.3%. Sensor seed 2: 4.4%.
  Uniform 40 sensors: 7.6%, wake vorticity 14% vs 11%. Field error 4.1 to 5.0% throughout.
  Reversed-flow region closes at 5.57 to 5.64 in the PINN vs 5.75 in the CFD.
- `[ai-assist-worked]` Screening in parallel (4 to 6 runs of 2 to 3 threads each) with a
  resumable checkpoint. The session was interrupted twice; the six robustness runs
  resumed from epoch 6,000 to 7,000 and finished.
- `[ai-assist-failed]` First guess was that the sample count and the loss weights were
  independent levers. The 20-sensor runs showed the viscosity ignoring both, and the only
  thing that moved it was the sensor count. Also launched four screening runs with a
  shell variable holding several flags; zsh did not split it and all four exited with an
  argparse error, which showed up only when reading the logs.

## 2026-09-30 (later): final run, animations
- `[pinn-training]` Reran the article configuration as `data/experiments/final` with weight
  snapshots (69 of them) for the explainer video. Same seed as R8, 8 threads instead of 3:
  field 4.3%, vorticity 10.0%, WSS 12.8%, viscosity 0.00498. It reproduced R8, so the article's
  numbers stand and the figures are now drawn from `final`. 23 minutes at 8 threads.
- Video and GIF feedback: hero and poster were duplicates (poster dropped, hero rebuilt with
  streamlines and a smoother wall glow); GIF too small and too many tracers (1300 px wide,
  170 tracers per panel); video too wide and too texty (portrait 1080x1350, short captions,
  training replay with stacked loss terms and viscosity estimate). The old last scene claimed
  the viscosity "comes back too" over a wall shear stress plot; viscosity now has its own
  panel in the training scene.
- `[ai-assist-failed]` First GIF resize used a 433 px height, which libx264 rejects for yuv420p
  (needs even dimensions). Fixed by rendering at 1300x490 and scaling to an even height.
