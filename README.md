# PINN Arterial Stenosis Flow

A physics-informed neural network (PINN) that reconstructs blood flow through a
stenosed (narrowed) vessel from sparse, noisy synthetic velocity readings, of
the kind a Doppler ultrasound exam would produce. It recovers near-wall
quantities that no sensor measures directly, wall shear stress in particular,
and infers the unknown blood viscosity as a learnable parameter.

Built as the code companion to a Towards Data Science article. The write-up in
progress is `article/part1_draft.md` (mechanism and the steady-flow build);
`article/dev-log.md` is the running build journal, including where an AI coding
assistant helped during development and where it did not.

## Project structure

- `src/pinn_bloodflow/`, the importable package. `geometry.py` holds the
  stenosis shape and the fluid/solid mask, `solver/` a from-scratch Chorin
  projection-method Navier-Stokes solver that generates the ground-truth data,
  `sensors/` the sparse noisy measurement sampling, `pinn/` the network,
  physics residuals and training loop, and `viz/` the plots and animations.
- `scripts/`, thin CLI entry points that call into the package.
  `evaluate.py` prints every number quoted in the article and regenerates the
  Phase 1 figures; `make_article_figures.py` builds the hero image and the
  cancelling-errors figure from the same checkpoint.
- `configs/`, YAML configs for Phase 1 (steady) and Phase 2 (pulsatile).
- `data/`, generated ground truth and sensor data. Gitignored; regenerate via
  the scripts.
- `tests/`, correctness tests for the solver and the PINN residuals.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Reproducing the Part 1 results

```bash
python scripts/generate_ground_truth.py --config configs/phase1_long.yaml --out data/ground_truth/phase1_long.npz   # long solve
python scripts/experiment.py --name final --set threads=3     # about 35 min on 3 CPU threads
python scripts/make_figures_v2.py --run data/experiments/final
```

Animations: `python scripts/make_animations.py` writes the looping
side-by-side tracer comparison, and `python scripts/make_explainer.py` writes
a 33 second portrait explainer (1080x1350, no audio, captions burned in) that
replays the loss curves, the viscosity estimate and the flow field as the
network trains. The explainer needs a run trained with `--set
snapshot_epochs=[...]`. Outputs go to `article/figures/`.

`experiment.py` trains one variant and writes its checkpoint and `metrics.json`
(velocity, vorticity, pressure, wall shear stress and viscosity errors, from
`src/pinn_bloodflow/evaluation.py`). `--sensors` picks the probe layout
(`random:N` or `zones:x0-x1:n,...`) and `--set key=value` overrides any `pinn`
setting. The older `configs/phase1_steady_v1.yaml` and `data/v1/` hold the
5H-channel version the first draft used.

## Status

Phase 1 (steady) and Phase 2 (pulsatile) are both built; see the dev log for
what each milestone covered and what broke along the way. Phase 1 was redone on
a 9H channel with a smoother wall shear stress reference for the rewritten
article (dev log, 2026-09-30). Phase 2 still uses the earlier 5H setup and has
not been redone.
