# Part 1: strict review findings

Independent editorial pass, plus my own verification. Tick items you want
me to act on, strike out ones you disagree with, add your own notes inline.

`[V]` = I re-derived it from the repo myself and confirmed it.
`[?]` = reported by the reviewer, not independently checked yet.

---

## BLOCKING

### 1. `[V]` The WSS peak agreement is two errors cancelling
`evaluate.py:105` computes the PINN's WSS with the *recovered* viscosity
(0.00609, 21.8% high). `compute_wss_ground_truth` uses the *true* one
(0.005). Verified numbers:

```
peak gt              0.3352
peak pinn @ nu_rec   0.3159   <- what the article reports
peak pinn @ nu_true  0.2594
near-wall gradient ratio pinn/gt = 0.774  (22.6% LOW)
rel L2: 0.244 @nu_rec,  0.220 @nu_true
```

The network's actual velocity gradient is 23% low. A 22%-high viscosity
multiplies it back up to something that looks like agreement. So the
sentence I added an hour ago, "what remains is mostly disagreement in the
shape of the rising limb rather than a systematic underestimate," is
backwards. There is a systematic underestimate.

It is the same amplitude-damping failure `dev-log.md:230` names as Phase
2's headline problem. Phase 1 has it too, hidden by the parameter error.

**Options:** (a) report both numbers and make the damping a finding that
sets up Part 2, or (b) also report WSS at the true viscosity as a
diagnostic. I lean (a); it is a genuine result and it links the two parts.

(So does this mean that the project itself did not work well? Then in might be needed to go back to the beginning to make it work well. At least the first article project should be "easy enough" since it's the basics more or less)

### 2. `[V]` The separation bubble never reattaches, and flow re-enters the outlet
Verified: near-wall `u < 0` continuously from `x = 2.19` to the outlet at
`x = 5.0`. Outlet column first five fluid cells:
`[-0.0044, -0.0198, -0.0341, -0.0457, -0.0538]`.

A zero-gradient / `p=0` outflow condition is not well posed with
re-entrant flow. The downstream half of the ground truth, and the WSS for
`x > 2.2`, inherit whatever the boundary imposes. Currently unmentioned
anywhere. The heatmap caption also calls it a bubble "just downstream of
the stenosis," which it is not.

**Note:** this is a limitation of the *ground truth*, not the PINN. Costs
one Limitations paragraph. A longer domain is the real fix, and that is a
re-run of everything.

(I do not really understand, but it does not seem important)

### 3. `[V]` Dynamic vs kinematic viscosity confusion
The article compares blood's dynamic viscosity (`3-4e-3 Pa·s`) against
the nondimensional kinematic `nu = 0.005` in the equations. Different
quantities. Blood's kinematic viscosity is ~3.3e-6 m²/s.

Second problem: the paragraph credits nondimensionalization with fixing
an optimizer-scaling issue, but 0.005 is itself ~2.3 orders from O(1), so
it fixes nothing there. The thing that actually helps is the log
parameterization, covered 30 lines later.

(Of course, fix this. And make sure you are using the same correct "tabulated" values everywhere)

### 4. `[V]` The `create_graph` failure mode is described backwards
Article says without it "`_grad(u_x, x)` gives you nothing, which kills
the viscous term," implying silent failure. Verified: it raises
`RuntimeError: element 0 of tensors does not require grad and does not
have a grad_fn` immediately. Loud, on epoch zero.

Also "Both flags" then lists three, and never mentions `retain_graph`.

(not really important. Even the original paragraph in the article might be superfluous. You tend to get caught up in explaining dumb details of the implementation)


### 5. `[?]` "It's the collocation count doing most of that work" is unmeasured
Two things changed at once (7x48 to 6x32, and 4000 to 1500 points).
Parameter count fell 2.6x, collocation count 2.7x, so on the arithmetic
they contributed about equally. The article turns this into advice.

**Decision needed:** run the 2-point ablation (two smoke tests, roughly
20 min), or cut the causal claim and the advice sentence.

(No idea)

### 6. `[?]` "From 20 noisy sensors" oversells what the sensors do
The network also gets 100 exact inlet velocities, 200 exact no-slip
points, and the exact geometry, all at weight 10, versus 20 noisy points
at weight 100. With an exact inlet, exact walls and the steady residual,
the field may be close to determined before any sensor is read.

**Decision needed:** run a zero-sensor ablation (data weight 0, nu fixed
at truth) and report what the sensors actually buy. One training run. If
the field error is still low without sensors, that is a significant
finding and the framing has to change.

(The walls no slip makes sense because it would be known, the others only if it really would be something known beforehand. Also, is blood not nonnewtonian?)

---

## SHOULD FIX

(It is too hard to understand what even you are saying on these. Just choose yourself based on reasonable criteria for our objective of writing educational articles)

- [ ] 7. `nu_true_hint` anchors the search within 4x of the answer, and
  the trajectory descends monotonically from it without ever crossing, so
  the init is load-bearing for the reported 21.8%. Article raises the
  objection and does not answer it.
- [ ] 8. "The loss has four terms" does not match the code: `train.py`
  builds three weighted groups, with the pressure gauge folded into `bc`
  at weight 10. Also unstated: there is no outlet BC at all; point counts;
  wall points are uniform in x, not arc length, so the steep flank where
  WSS peaks gets no extra resolution.
- [ ] 9. WSS is computed on the bottom wall only, which undercuts the
  stated reason for choosing an eccentric geometry ("different shear on
  the two walls"). Either show the top wall or drop that justification.
- [ ] 10. The two WSS curves are not the same measurement: PINN evaluated
  on the analytic wall, ground truth a two-cell difference at their
  midpoint. Both use `nu du/dy`, which is only the true wall shear at the
  crest, not on the inclined flanks, which is exactly where the residual
  disagreement is.
- [ ] 11. "7% noise" is 7% of *mean inlet velocity*, not of local speed.
  Median relative noise at the actual probes is 5.1%, under 3.5% in the
  throat.
- [ ] 12. Sensor placement is framed as a deliberate handicap, but Part 2
  root-causes uniform placement as the bottleneck and reverses it. Phase 1
  has 1 of 20 probes near the throat, where 88% of the WSS signal is.
- [ ] 13. The numbers in my new WSS paragraph are correct but exist only
  in the draft, not in any script or the dev-log. "Smoothed" is also
  window-dependent and unstated (k=9 gives 21.4%, k=21 gives 22.4%).
- [ ] 14. The PDE loss is still falling in a straight line at epoch
  15,000, two orders above the other terms. The most obvious remaining
  lever is more epochs, which contradicts "the lever with the most left in
  it is sensor count."
- [ ] 15. Pressure is promised as an inferred output and never reported.
- [ ] 16. The reader never learns how collocation or boundary points are
  sampled, or that boundary points are fixed while collocation resamples.
  Also the sensor margin is a vertical gap, not a perpendicular distance.
- [ ] 17. Voice regressions against our own tic list: four "worth
  [gerund]-ing" transitions, "genuine bugs", "genuinely contested". And
  the em-dash removal converted asides into nine "rather than" clauses,
  which is becoming the new signature.
- [ ] 18. The closing sentence of the bug story parses as the PINN being
  off by 50x. It was the reference calculation.

---

## MINOR

- [ ] 19. Caption fixes: sensors marked on bottom panel only; `|velocity|`
  cannot show recirculation, only a low-speed band; there is a third
  error hot band on the inlet centerline that goes unmentioned; heatmap
  colorbar is absolute error, which flatters the near-wall region the
  article claims is hard.
- [ ] 20. "roughly four times water" takes the top of the stated 3-4
  range. "Killed at 55 minutes" vs dev-log's "timed out". The 7-10 min
  figure is an 8,000-epoch smoke test at old settings; every number in the
  article comes from a 15,000-epoch run that was never timed. No range
  quoted from [4] for the 21.8% comparison. Parameter count (~5,500) not
  given. WSS units not stated as nondimensional.
- [ ] 21. Pacing: two paragraphs of third-person prose before the first
  "I"; equations given twice in two notations; compute budget discussed
  before the reader knows what collocation points do; four consecutive
  "[constraint], so [consequence]" sentences in Limitations.

---

## Reviewer's one positive note

The paragraph about the parameter being twice as accurate at epoch 8,200
as at the end, with no principled stopping criterion offered, was called
the strongest writing in the piece and the register the rest should match.

---

## STATUS (updated as I work)

**Done, in the article:**
- 3 viscosity units. Consistent values everywhere now: dynamic 3.5e-3
  Pa.s, density 1060 kg/m3, kinematic 3.3e-6 m2/s. The false claim that
  nondimensionalization fixes optimizer scaling is gone; that argument
  moved to the log parameterization, where it is true. Re=200 now states
  what it physically means (0.11 m/s in a 6 mm carotid, slow end of
  physiological).
- 4 create_graph. Cut to three lines instead of corrected, per your note
  about implementation minutiae.
- 5 collocation ablation. Not running it. The causal claim and the advice
  built on it are deleted; the paragraph says plainly that I changed two
  things at once and cannot attribute the speedup.
- 1 WSS cancelling errors. Article reports the gradient and the viscosity
  separately and says the product agreement was coincidental. Numbers
  refresh after the retrain.
- 2 outflow. New Limitations paragraph, in plain language.
- 6 (partial) non-Newtonian, answered in Limitations, including why the
  assumption is weakest exactly where the domain is also weakest.
- 7 nu_true_hint anchoring. Acknowledged, with the honest alternative.
- 8 loss terms. Three weighted groups, gauge folded into bc, point counts
  given, absent outlet condition stated.
- 9 eccentric geometry. No longer promises a two-wall comparison the
  article does not show.
- 10 WSS approximation caveats. Both stated.
- 11 7% noise. Now: median 4.6% relative to local speed, 2.8% in the
  fastest flow, 56.6% at one probe in nearly stagnant fluid.
- 12 sensor placement. States 1 of 20 near the throat, and flags that
  Part 2 reverses this framing.
- 15 pressure. Reported: 12.8% gauge-shifted error, pressure drop
  recovered to about 4%, which is the clinically meaningful one.
- 16 sampling. Collocation and boundary sampling described, including
  that boundary points are fixed and the margin is vertical, not
  perpendicular.
- 17 voice. worth/genuine removed; "rather than" down from 13 to 9.
- 18 misparsing sentence. Rewritten.
- 20 (partial) parameter count added; timing replaced with a measured
  0.129 s/epoch; "killed at 55 minutes" now says I gave up on it.
- 21 (partial) equations no longer read as repeated.

**Done, in the code:**
- `evaluate.py` reports near-wall du/dy separately from WSS, plus WSS at
  the true viscosity, the error decomposition, and pressure. Every number
  quoted in the article now comes from script output (item 13).
- `_train_phase1_chunk.py`: resumable, atomic checkpoints every 2000
  epochs, with `--freeze-nu` and `--data-weight` for the ablation.

**Humanizer + guidelines pass (2026-09-17):**
- 17 regression. Three "worth X" transitions were back ("worth its own
  sentence", "Worth being precise about that", "Pressure is worth a
  moment"). All three cut, the content kept. Count is now zero.
- Staging phrase "for a reason that matters more than it looks" cut from
  the tanh paragraph; the reason follows it anyway.
- Third-person slip: "the article has been promising it" became "I
  promised pressure as an inferred quantity."
- Grammar bug in the BC paragraph, from an earlier patch: "Though they
  are spaced uniformly in `x`, not along the curve, so the steep flank
  ..." had no main clause. Rewritten.
- 14 answered without reporting the longer run, since the 15k history
  proves it on its own: the PDE term falls another 31% over the last
  quarter and ends ~140x above the data term. The claim "the lever with
  the most left in it is sensor count, not optimizer tuning" was wrong
  and is replaced by a statement that 15,000 epochs did not converge.
- Precision: 0.766 is the gradient ratio at the *ground truth's* peak
  station, while 0.316/0.259 are the PINN's *own* peak. The sentence now
  says which is which, and quotes the 22% whole-wall gradient error too.
- README had 7 em-dashes, never caught because `check_dashes.sh`
  defaulted to `article/*.md` only. README rewritten dash-free and added
  to the checker's default file list. README also pointed at draft.md and
  claimed both phases were in progress; both fixed.
- draft.md marked SUPERSEDED FOR PART 1 in its header, retained as Part 2
  source for sections 9 to 11.

**Still open after that pass:**
- Kicker density. Four of eight sections end on a punch and two still
  share the same A/not-A shape ("Both framings are defensible. Only one
  of them survived contact with the harder problem." / "The solution was
  fine. The thing measuring it was not."). Guidelines say keep one where
  it is earned. Pick one to flatten.
- "rather than" holds at 8 in reader-facing prose. Each survives the
  diagnostic individually; the density is the question.
- Bold lead-ins on Data / PDE / BC. Kept, because those labels are the
  actual config keys and carry information, but it is the one place the
  piece uses a template shape.

**Not doing (decided 2026-09-17):**
- The 30,000-epoch run stays out of the article. Piece is introductory
  and the extra run does not earn its space. For the record, it lands at
  nu 13.0%, field 4.0%, WSS 27.5%, gradient 23.6%.

**Was waiting on the 30,000-epoch run:**
- 14 whether the result was simply undertrained. The trajectory is
  identical to the old run through epoch 15,000 (same seed), so the open
  question is only what happens after. The viscosity was drifting back
  upward when the old run stopped, so the field and the gradient may
  improve while the parameter estimate gets worse.
- 6 the zero-sensor ablation, queued behind it.
- 19 captions, once figures regenerate.

---

## Reach-focused finalization (2026-09-29)

Three independent reviews (title/SEO, reader engagement, figures) fed a rewrite.

**Done:**
- Title "How to Use a PINN for a Navier-Stokes Inverse Problem"; PyTorch and the
  concrete task moved to the subtitle. Primary keyword also in the first paragraph.
- Part 1 cut from about 3,250 to about 2,400 prose words (roughly 10.5 min). One
  arc: residual, learnable viscosity, results, cancelling errors. The long version
  is kept in `part1_long_version.md` as source for later articles (architecture and
  loss detail, sensor design, reference-calculation bug, error heatmap, raw WSS plot).
- Item 6 closed: zero-sensor ablation (data weight 0, nu frozen at truth, 15,000
  epochs, one seed): field 26.4%, v 51.1%, pressure 55.5%, WSS 52.9%, gradient
  ratio at peak 0.494. With sensors: 4.7%, 29%, 12.8%, 24.4%, 0.766. The sensors
  carry a large share of the answer, so the framing stands. In the article's
  Limitations.
- Item 19 closed: all captions rewritten; sensors on both panels; red u=0 contour
  supports the recirculation claim; relative-error panel added.
- New scripts/make_article_figures.py (hero, no text; cancelling-errors figure).
  A figure-review claim that the PINN peak is 0.313 was checked and is wrong: 0.316 is
  the PINN's own maximum, 0.313 is its value at the ground truth's peak station.

**Known quirk:** `evaluate.py` crashes at the plotting step when given the
frozen-nu ablation checkpoint (degenerate figure height in one plot). All numbers
print before it. Main-run figures are unaffected.

**Still open:** `[repo link]`; final pick of SEO title vs. alternatives.
