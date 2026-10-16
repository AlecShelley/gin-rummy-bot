import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Train the v5 Q-net by regression on the search's per-candidate EV tables.

Target parameterization (the load-bearing choice): the net regresses the
PAIRED DIFF vs the teacher candidate, in points --

    y[i, a] = 100 * (ev[i, a] - ev[i, t_idx])

-- because (a) the stored per-candidate `se` is the SE of exactly this
quantity (the paired design cancels between-world variance, so diff labels
are far cleaner than absolute EVs); (b) ranking, hence play, is invariant to
the per-state constant; (c) symmetric world-model bias cancels in the diff
(lesson #9's mechanism working for us); (d) y[t_idx] = 0 exactly -- a free
noise-free anchor. Absolute-EV regression is what blurred the 1-3 pt
candidate gaps in the value-leaf failure (lesson #8). `--target ev` keeps
the ablation available.

Loss: precision-weighted MSE over the labeled candidates only (mask
everything else), w = 1/max(se_pts, 1.0)^2 -- the floor caps the se=0
teacher anchor and stops lucky low-SE rows from dominating.

Recipe follows value_train.py: whole-SHARD val split (states within a hand
share history -- value run 1 overfit in ONE epoch without this), obs kept
float16 on CPU, minibatch -> GPU cast in forward, AdamW + cosine + dropout +
early stop, strip_dropout before save.

Offline gates (printed on the best checkpoint; thresholds from the plan):
top-1 agreement (raw / confident), pairwise ranking accuracy on confident
pairs, diff RMSE vs mean label SE, and THE stop gate -- on deviation states,
the fraction where Q[chosen] > Q[t_idx]. Below ~60% => stop and rethink
(that is where the search's value lives; the v3-clone failure repeating).

    .venv/Scripts/python.exe qtrain.py --data results/v5_q/gen \
        --out results/v5_q/qnet.pt
"""

import argparse
import glob
import time

import numpy as np

from legacy.qnet import NUM_ACTIONS, save_model

CMAX = 8
SE_FLOOR_PTS = 1.0


def load_shards(prefixes, val_frac=0.08, seed=7):
    files = []
    for prefix in prefixes:
        files.extend(sorted(glob.glob(f"{prefix}_*.npz")))
    assert files, f"no shards match {prefixes}"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_frac))
    val_set = {files[i] for i in order[:n_val]}

    def stack(fs):
        keys = ("obs", "cand", "ev", "se", "t_idx", "n_cand", "chosen",
                "reason")
        cols = {k: [] for k in keys}
        for f in fs:
            d = np.load(f)
            if len(d["t_idx"]) == 0:
                continue
            for k in keys:
                cols[k].append(d[k])
        return {k: np.concatenate(v) for k, v in cols.items()}

    train = stack([f for f in files if f not in val_set])
    val = stack(sorted(val_set))
    print(f"{len(files)} shards -> train {len(train['t_idx'])} / "
          f"val {len(val['t_idx'])} states "
          f"(val deviation rate {(val['reason'] == 1).mean():.3f})",
          flush=True)
    return train, val


def make_targets(d, target="diff"):
    """-> (cand_idx int64 (N,8) pad->0, y float32 (N,8) pts, w float32 (N,8)
    zero on pads, mask bool (N,8))."""
    n = len(d["t_idx"])
    rows = np.arange(n)
    n_cand = d["n_cand"].astype(np.int64)
    mask = np.arange(CMAX)[None, :] < n_cand[:, None]
    ev_pts = d["ev"].astype(np.float32) * 100.0
    if target == "diff":
        y = ev_pts - ev_pts[rows, d["t_idx"].astype(np.int64)][:, None]
    else:
        y = ev_pts
    se_pts = d["se"].astype(np.float32) * 100.0
    w = mask / np.maximum(se_pts, SE_FLOOR_PTS) ** 2
    cand_idx = np.where(mask, d["cand"], 0).astype(np.int64)
    return cand_idx, (y * mask).astype(np.float32), w.astype(np.float32), mask


def gate_metrics(q_cand, d, mask):
    """Offline gates on numpy arrays. q_cand: net outputs gathered at the
    candidate ids, (N,8), pads already at -inf."""
    n = len(d["t_idx"])
    rows = np.arange(n)
    t_idx = d["t_idx"].astype(np.int64)
    chosen = d["chosen"].astype(np.int64)
    y = d["ev"].astype(np.float32) * 100.0
    y = y - y[rows, t_idx][:, None]
    se = d["se"].astype(np.float32) * 100.0
    y_m = np.where(mask, y, -np.inf)

    net_top = q_cand.argmax(1)
    lab_top = y_m.argmax(1)
    out = {"top1_raw": float((net_top == lab_top).mean())}

    # confident states: label best-vs-second gap > 2 * max of their SEs
    part = np.argsort(-y_m, axis=1)
    b1, b2 = part[:, 0], part[:, 1]
    gap = y[rows, b1] - y[rows, b2]
    conf = gap > 2.0 * np.maximum(se[rows, b1], se[rows, b2])
    out["conf_frac"] = float(conf.mean())
    out["top1_conf"] = float((net_top == lab_top)[conf].mean())

    # pairwise ranking on confident pairs (pads zeroed to avoid inf-inf NaNs;
    # the valid mask excludes them anyway)
    q_fin = np.where(mask, q_cand, 0.0)
    dy = y[:, :, None] - y[:, None, :]
    dq = q_fin[:, :, None] - q_fin[:, None, :]
    pair_se = np.maximum(se[:, :, None], se[:, None, :])
    valid = mask[:, :, None] & mask[:, None, :] & (dy > 2.0 * pair_se)
    out["pair_rank"] = float((dq[valid] > 0).mean())

    # diff RMSE (non-teacher candidates) vs the labels' own SE
    nt = mask.copy()
    nt[rows, t_idx] = False
    pred_diff = q_cand - q_cand[rows, t_idx][:, None]
    out["diff_rmse"] = float(np.sqrt(((pred_diff - y)[nt] ** 2).mean()))
    out["label_se_mean"] = float(se[nt].mean())

    # THE stop gate: deviation states
    dev = d["reason"] == 1
    out["dev_frac"] = float(dev.mean())
    if dev.any():
        qd = q_cand[dev]
        out["dev_beats_prior"] = float(
            (qd[np.arange(dev.sum()), chosen[dev]]
             > qd[np.arange(dev.sum()), t_idx[dev]]).mean())
        out["dev_top1"] = float((qd.argmax(1) == chosen[dev]).mean())
    return out


def build_model_dropout(hidden, dropout):
    import torch.nn as nn
    from legacy.qnet import INPUT_DIM
    layers, prev = [], INPUT_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, NUM_ACTIONS))
    return nn.Sequential(*layers)


def strip_dropout(model, hidden):
    from legacy.qnet import build_model
    clean = build_model(hidden)
    src = [m for m in model if m.__class__.__name__ == "Linear"]
    dst = [m for m in clean if m.__class__.__name__ == "Linear"]
    for a, b in zip(src, dst):
        b.load_state_dict(a.state_dict())
    return clean


def train(data_prefixes, out_path, target="diff", hidden=(512, 512, 256),
          epochs=40, batch=8192, lr=3e-4, weight_decay=1e-4, dropout=0.2,
          patience=5, seed=7):
    import torch
    tr, va = load_shards(data_prefixes, seed=seed)
    ci_tr, y_tr_np, w_tr_np, _ = make_targets(tr, target)
    ci_va, y_va_np, w_va_np, mask_va = make_targets(va, target)
    base = float((w_va_np * y_va_np ** 2).sum() / w_va_np.sum())
    print(f"baseline (predict-0 = always-teacher) weighted val MSE: "
          f"{base:.4f} pts^2", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = build_model_dropout(hidden, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    obs_tr = torch.from_numpy(tr["obs"])            # float16, stays on CPU
    ci_tr = torch.from_numpy(ci_tr)
    y_tr = torch.from_numpy(y_tr_np)
    w_tr = torch.from_numpy(w_tr_np)
    obs_va = torch.from_numpy(va["obs"]).to(dev)
    ci_va_t = torch.from_numpy(ci_va).to(dev)
    y_va = torch.from_numpy(y_va_np).to(dev)
    w_va = torch.from_numpy(w_va_np).to(dev)

    def forward_q(obs, ci):
        x = obs.float().reshape(len(obs), -1)
        return model(x).gather(1, ci)

    n = len(y_tr)
    best_val, best_state, best_ep, since = 1e18, None, -1, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total_l, total_w = 0.0, 0.0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ob = obs_tr[idx].to(dev, non_blocking=True)
            cb = ci_tr[idx].to(dev, non_blocking=True)
            yb = y_tr[idx].to(dev, non_blocking=True)
            wb = w_tr[idx].to(dev, non_blocking=True)
            q = forward_q(ob, cb)
            loss = (wb * (q - yb) ** 2).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_l += float((wb * (q.detach() - yb) ** 2).sum())
            total_w += float(wb.sum())
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(obs_va), 65536):
                chunks.append(forward_q(obs_va[i:i + 65536],
                                        ci_va_t[i:i + 65536]))
            qv = torch.cat(chunks)
            val_mse = float(((w_va * (qv - y_va) ** 2).sum()
                             / w_va.sum()))
        marker = ""
        if val_mse < best_val:
            best_val, best_ep, since = val_mse, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = " *"
        else:
            since += 1
        print(f"epoch {ep:2d}  train {total_l / total_w:.4f}  "
              f"val {val_mse:.4f}  ({time.time() - t0:.0f}s){marker}",
              flush=True)
        if since >= patience:
            print(f"early stop (no val improvement for {patience} epochs)",
                  flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(obs_va), 65536):
            chunks.append(forward_q(obs_va[i:i + 65536],
                                    ci_va_t[i:i + 65536]))
        q_cand = torch.cat(chunks).cpu().numpy()
    q_cand = np.where(mask_va, q_cand, -np.inf)
    gates = gate_metrics(q_cand, va, mask_va)

    clean = strip_dropout(model.cpu(), hidden)
    save_model(clean, hidden, out_path, target=target)
    print(f"best val MSE {best_val:.4f} (epoch {best_ep}) vs baseline "
          f"{base:.4f} ({1 - best_val / base:+.1%})  -> {out_path}",
          flush=True)
    print("offline gates:", {k: round(v, 4) for k, v in gates.items()},
          flush=True)
    dev_gate = gates.get("dev_beats_prior", 0.0)
    print(f"STOP GATE (dev_beats_prior >= 0.60): "
          f"{'PASS' if dev_gate >= 0.60 else 'FAIL -- stop and rethink'} "
          f"({dev_gate:.3f})", flush=True)
    print(f"RMSE GATE (diff_rmse <= label_se_mean): "
          f"{'PASS' if gates['diff_rmse'] <= gates['label_se_mean'] else 'FAIL'} "
          f"({gates['diff_rmse']:.2f} vs {gates['label_se_mean']:.2f})",
          flush=True)
    return best_val, gates


def main():
    ap = argparse.ArgumentParser("v5 Q-net training")
    ap.add_argument("--data", default="results/v5_q/gen")
    ap.add_argument("--out", default="results/v5_q/qnet.pt")
    ap.add_argument("--target", choices=("diff", "ev"), default="diff")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden", default="512,512,256")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    train(args.data.split(","), args.out, args.target, hidden, args.epochs,
          args.batch, args.lr, args.weight_decay, args.dropout,
          args.patience, args.seed)


if __name__ == "__main__":
    main()
