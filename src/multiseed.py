"""Multi-seed robustness: train several seeds and record sigma_min^+(J) and test
accuracy per snapshot, so the "no dip at the transition" result can be checked
across seeds. Writes multiseed.csv (seed, step, test_acc, sigma_min).

This is intentionally self-contained (trains and diagnoses in one pass) so it
can be run without first producing per-seed run files.
"""
import argparse, csv, os, time, torch, torch.nn as nn
from torch.func import functional_call, jacrev
from train import build_data, build_model


def run_seed(seed, cfg):
    Xtr, Ytr, Ytr_c, Xte, Yte_c = build_data(cfg.p, cfg.train_frac, cfg.split_seed)
    model = build_model(cfg.p, cfg.width, cfg.init_scale, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pnames = [n for n, _ in model.named_parameters()]
    R = Ytr.numel()

    def resid(pd):
        return (functional_call(model, pd, (Xtr,)) - Ytr).reshape(-1)

    snaps, steps = [], []
    for s in range(cfg.steps + 1):
        opt.zero_grad()
        (((model(Xtr) - Ytr) ** 2).mean()).backward()
        opt.step()
        if s % cfg.snapshot_every == 0:
            snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            steps.append(s)
    out = []
    for st, sd in zip(steps, snaps):
        model.load_state_dict(sd)
        with torch.no_grad():
            te = (model(Xte).argmax(1) == Yte_c).float().mean().item()
        J = jacrev(resid)({k: v for k, v in sd.items()})
        Jm = torch.cat([J[n].reshape(R, -1) for n in pnames], dim=1).float()
        out.append((st, te, torch.linalg.svdvals(Jm).min().item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--p", type=int, default=11)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=2.0)
    ap.add_argument("--init_scale", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--snapshot_every", type=int, default=1000)
    ap.add_argument("--out", default="results")
    cfg = ap.parse_args()

    os.makedirs(cfg.out, exist_ok=True)
    path = os.path.join(cfg.out, "multiseed.csv")
    t0 = time.time()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "test_acc", "sigma_min"])
        for sd in cfg.seeds:
            for st, te, sm in run_seed(sd, cfg):
                w.writerow([sd, st, te, sm])
            f.flush()
            print(f"seed {sd} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
