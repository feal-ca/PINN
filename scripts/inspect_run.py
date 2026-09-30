"""Quick look at a trained variant: velocity magnitude, vorticity, and errors."""
import sys
from pathlib import Path
import numpy as np, torch, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn_bloodflow.evaluation import evaluate
from pinn_bloodflow.geometry import StenosisParams
from pinn_bloodflow.pinn.train import build_model

run = Path(sys.argv[1]); out = sys.argv[2]
gt = np.load(sys.argv[3] if len(sys.argv) > 3 else "data/ground_truth/phase1_long.npz")
ck = torch.load(run / "model.pt", weights_only=False)
cfg = yaml.safe_load(open("configs/phase1_long.yaml")); g = cfg["geometry"]
params = StenosisParams(H=g["H"], Lx=g["Lx"], x0=g["stenosis_x0"], pct_ds=g["stenosis_pct_ds"], sigma=g["stenosis_sigma"], eccentric=g["eccentric"])
model = build_model(ck["cfg"], params); model.load_state_dict(ck["model_state"]); model.eval()
m, ex = evaluate(model, ck["log_nu"].item(), gt, params)
mask = gt["mask_p"]; xp, yp = gt["xp"], gt["yp"]
ext = [0, params.Lx, 0, params.H]
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
mag_t = np.ma.array(np.hypot(gt["u"], gt["v"]), mask=~mask); mag_p = np.ma.array(np.hypot(ex["u_p"], ex["v_p"]), mask=~mask)
for a, f, t in zip(ax[:2], [mag_t, mag_p], ["CFD |u|", "PINN |u|"]):
    a.imshow(f, origin="lower", extent=ext, vmin=0, vmax=2.5); a.set_title(t)
wt = np.ma.array(ex["w_t"], mask=~mask); wp = np.ma.array(ex["w_p"], mask=~mask)
for a, f, t in zip(ax[2:], [wt, wp], ["CFD vorticity", "PINN vorticity"]):
    a.imshow(f, origin="lower", extent=ext, vmin=-20, vmax=20, cmap="RdBu_r"); a.set_title(t)
ax[3].plot(ck["sensors"]["x"], ck["sensors"]["y"], "ko", ms=4)
plt.tight_layout(); plt.savefig(out, dpi=60); print(vars(m))
