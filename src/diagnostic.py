"""Compute the normal-hyperbolicity diagnostic on training snapshots.

For each snapshot we form the residual Jacobian J = d r / d theta on the
training set, where r(theta) = (f(theta; x_i) - y_i)_i. At an interpolating
point, the row space of J is the normal space of the zero-loss manifold and
ker(J) is its tangent space; the squared singular values of J are the nonzero
eigenvalues of the Gauss-Newton Hessian. The smallest nonzero singular value
sigma_min^+(J) is therefore the smallest normal restoring curvature. A loss of
normal hyperbolicity (a fold / bifurcation of the slow manifold) corresponds to
sigma_min^+(J) -> 0.

Output: a tidy CSV with one row per evaluated snapshot.
"""
import argparse, csv, os, torch, torch.nn as nn
from torch.func import functional_call, jacrev

from train import build_model  # reuse the exact architecture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="path to run_seed*.pt")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th snapshot")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    d = torch.load(args.run, weights_only=False)
    cfg = d["cfg"]
    Xtr, Ytr, Ytr_c, Xte, Yte_c = d["Xtr"], d["Ytr"], d["Ytr_c"], d["Xte"], d["Yte_c"]

    model = build_model(cfg["p"], cfg["width"], cfg["init_scale"], cfg["seed"])
    pnames = [n for n, _ in model.named_parameters()]
    R = Ytr.numel()

    def resid(pd):
        return (functional_call(model, pd, (Xtr,)) - Ytr).reshape(-1)

    def accs(sd):
        model.load_state_dict(sd)
        with torch.no_grad():
            tr = (model(Xtr).argmax(1) == Ytr_c).float().mean().item()
            te = (model(Xte).argmax(1) == Yte_c).float().mean().item()
            mse = ((model(Xtr) - Ytr) ** 2).mean().item()
            wn = sum((q * q).sum().item() for q in model.parameters()) ** 0.5
        return tr, te, mse, wn

    os.makedirs(args.out, exist_ok=True)
    seed = cfg["seed"]
    out_csv = os.path.join(args.out, f"results_seed{seed}.csv")
    rows = []
    for i, (st, sd) in enumerate(zip(d["snap_steps"], d["snaps"])):
        if i % args.stride:
            continue
        J = jacrev(resid)({k: v for k, v in sd.items()})
        Jm = torch.cat([J[n].reshape(R, -1) for n in pnames], dim=1).float()
        smin = torch.linalg.svdvals(Jm).min().item()
        tr, te, mse, wn = accs(sd)
        rows.append((st, tr, te, mse, wn, smin))
        print(f"step {st:>6}  teAcc {te:.3f}  sigma_min+ {smin:.4e}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_acc", "test_acc", "train_mse", "wnorm", "sigma_min_plus"])
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
