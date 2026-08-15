"""Train a two-layer ReLU network to grok modular addition (squared loss) and
snapshot parameters across training. Snapshots let the normal-hyperbolicity
diagnostic (diagnostic.py) run in a separate pass without slowing training.

Defaults reproduce the paper run: a clear memorization plateau followed by a
delayed generalization transition.
"""
import argparse, os, torch, torch.nn as nn


def build_data(p, train_frac, split_seed=1):
    ab = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    X = torch.zeros(len(ab), 2 * p)
    X[torch.arange(len(ab)), ab[:, 0]] = 1
    X[torch.arange(len(ab)), p + ab[:, 1]] = 1
    Yc = (ab[:, 0] + ab[:, 1]) % p
    Y = torch.zeros(len(ab), p)
    Y[torch.arange(len(ab)), Yc] = 1
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(ab), generator=g)
    ntr = int(train_frac * len(ab))
    tr, te = perm[:ntr], perm[ntr:]
    return X[tr], Y[tr], Yc[tr], X[te], Yc[te]


def build_model(p, H, init_scale, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2 * p, H), nn.ReLU(), nn.Linear(H, p, bias=False))
    with torch.no_grad():
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight *= init_scale
                if m.bias is not None:
                    m.bias.zero_()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=11)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=2.0)
    ap.add_argument("--init_scale", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--snapshot_every", type=int, default=500)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    Xtr, Ytr, Ytr_c, Xte, Yte_c = build_data(args.p, args.train_frac, args.split_seed)
    model = build_model(args.p, args.width, args.init_scale, args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def acc(Xs, yc):
        with torch.no_grad():
            return (model(Xs).argmax(1) == yc).float().mean().item()

    def wnorm():
        with torch.no_grad():
            return sum((q * q).sum().item() for q in model.parameters()) ** 0.5

    log, snaps, snap_steps = [], [], []
    for s in range(args.steps + 1):
        opt.zero_grad()
        (((model(Xtr) - Ytr) ** 2).mean()).backward()
        opt.step()
        if s % args.log_every == 0:
            log.append((s, ((model(Xtr) - Ytr) ** 2).mean().item(),
                        acc(Xtr, Ytr_c), acc(Xte, Yte_c), wnorm()))
        if s % args.snapshot_every == 0:
            snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            snap_steps.append(s)

    path = os.path.join(args.out, f"run_seed{args.seed}.pt")
    torch.save({"log": log, "snaps": snaps, "snap_steps": snap_steps,
                "Xtr": Xtr, "Ytr": Ytr, "Ytr_c": Ytr_c, "Xte": Xte, "Yte_c": Yte_c,
                "cfg": vars(args)}, path)
    print(f"saved {path}  final={log[-1]}  n_snapshots={len(snaps)}")


if __name__ == "__main__":
    main()
