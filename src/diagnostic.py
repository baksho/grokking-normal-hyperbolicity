"""Normal-hyperbolicity diagnostic on training snapshots.

For each snapshot we form the residual Jacobian J = d r / d theta on the
training set. At interpolation, row(J) is the manifold's normal space and
ker(J) its tangent space; the squared singular values of J are the nonzero
Gauss-Newton eigenvalues, so the k smallest singular values are the k slowest
normal restoring rates. A loss of normal hyperbolicity is sigma_min^+(J) -> 0.

Writes results_seed{seed}.csv with the k smallest singular values per snapshot.
"""
import argparse, csv, os, torch, torch.nn as nn
from torch.func import functional_call, jacrev
from train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to run_seed*.pt")
    ap.add_argument("--k", type=int, default=6, help="number of smallest singular values")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    d = torch.load(args.run, weights_only=False)
    cfg = d["cfg"]
    Xtr, Ytr, Ytr_c, Xte, Yte_c = d["Xtr"], d["Ytr"], d["Ytr_c"], d["Xte"], d["Yte_c"]
    model = build_model(cfg["p"], cfg["width"], cfg["init_scale"], cfg["seed"])
    pnames = [n for n, _ in model.named_parameters()]
    R = Ytr.numel()

    def resid(pd):
        return (functional_call(model, pd, (Xtr,)) - Ytr).reshape(-1)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"results_seed{cfg['seed']}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_acc", "test_acc", "train_mse", "wnorm"]
                   + [f"s{i+1}" for i in range(args.k)])
        for st, sd in zip(d["snap_steps"], d["snaps"]):
            model.load_state_dict(sd)
            with torch.no_grad():
                tr = (model(Xtr).argmax(1) == Ytr_c).float().mean().item()
                te = (model(Xte).argmax(1) == Yte_c).float().mean().item()
                mse = ((model(Xtr) - Ytr) ** 2).mean().item()
                wn = sum((q * q).sum().item() for q in model.parameters()) ** 0.5
            J = jacrev(resid)({k: v for k, v in sd.items()})
            Jm = torch.cat([J[n].reshape(R, -1) for n in pnames], dim=1).float()
            sv = torch.linalg.svdvals(Jm)
            ksmall = sv[-args.k:].flip(0).tolist()      # smallest, 2nd smallest, ...
            w.writerow([st, tr, te, mse, wn] + ksmall)
            print(f"step {st:6d}  teAcc {te:.3f}  sigma_min+ {ksmall[0]:.4e}", flush=True)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
