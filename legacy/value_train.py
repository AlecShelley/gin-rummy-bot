import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Train the value leaf by regression on playout outcomes (roadmap #1).

Loads value_gen shards (comma-separated prefixes), splits train/val BY SHARD
(states from one hand never straddle the split), fits the MLP on GPU, and
reports val MSE against three baselines: predict-0, the best linear rescale
of the handcrafted leaf, and a linear model on all 8 scalars. The net has to
beat all three to matter.

Memory: planes stay uint8 in RAM; each batch is expanded to float32 on the
GPU. 15M states ~ 5GB.

Run 1 lesson (results/v4_value/train_run1.log): 5.4M states from only 300k
hands overfit after ONE epoch (states within a hand share one noisy outcome).
Cure: more hands, sparser per-hand recording, dropout + weight decay, early
stop on patience.

    .venv/Scripts/python.exe value_train.py \
        --data results/v4_value/train,results/v4_value/train2 \
        --out results/v4_value/value.pt
"""

import argparse
import glob
import time

import numpy as np

from legacy.valuenet import INPUT_DIM, save_model


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
        planes, scalars, labels = [], [], []
        for f in fs:
            d = np.load(f)
            planes.append(d["planes"])
            scalars.append(d["scalars"])
            labels.append(d["labels"])
        return (np.concatenate(planes), np.concatenate(scalars),
                np.concatenate(labels).astype(np.float32))

    train = stack([f for f in files if f not in val_set])
    val = stack(sorted(val_set))
    print(f"{len(files)} shards -> train {len(train[2])} / val {len(val[2])} "
          f"states", flush=True)
    return train, val


def baselines(train, val):
    """Val MSE of: zero, scaled leaf scalar, linear on the 8 scalars."""
    _, s_tr, y_tr = train
    _, s_va, y_va = val
    out = {"zero": float((y_va ** 2).mean())}
    leaf_tr, leaf_va = s_tr[:, 6], s_va[:, 6]
    a = float((leaf_tr * y_tr).sum() / max((leaf_tr ** 2).sum(), 1e-9))
    out["scaled_leaf"] = float(((y_va - a * leaf_va) ** 2).mean())
    out["leaf_scale_a"] = a
    a_tr = np.concatenate([s_tr, np.ones((len(s_tr), 1), np.float32)], axis=1)
    a_va = np.concatenate([s_va, np.ones((len(s_va), 1), np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
    out["linear_scalars"] = float(((y_va - a_va @ w) ** 2).mean())
    return out


def build_model_dropout(hidden, dropout):
    import torch.nn as nn
    layers, prev = [], INPUT_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def strip_dropout(model, hidden):
    """Copy weights into the dropout-free architecture valuenet expects."""
    from legacy.valuenet import build_model
    clean = build_model(hidden)
    src = [m for m in model if m.__class__.__name__ == "Linear"]
    dst = [m for m in clean if m.__class__.__name__ == "Linear"]
    for a, b in zip(src, dst):
        b.load_state_dict(a.state_dict())
    return clean


def train(data_prefixes, out_path, hidden=(512, 512, 256), epochs=40,
          batch=8192, lr=3e-4, weight_decay=1e-4, dropout=0.2, patience=5,
          seed=7):
    import torch
    tr, va = load_shards(data_prefixes, seed=seed)
    base = baselines(tr, va)
    print("baseline val MSE:", {k: (f"{v:.5f}" if isinstance(v, float) else v)
                                for k, v in base.items()}, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = build_model_dropout(hidden, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    p_tr = torch.from_numpy(tr[0])       # uint8 (N,6,52)
    s_tr = torch.from_numpy(tr[1])
    y_tr = torch.from_numpy(tr[2])
    p_va = torch.from_numpy(va[0]).to(dev)
    s_va = torch.from_numpy(va[1]).to(dev)
    y_va = torch.from_numpy(va[2]).to(dev)

    def forward(p, s):
        x = torch.cat([p.float().reshape(len(p), -1), s], dim=1)
        return model(x).squeeze(1)

    n = len(y_tr)
    best_val, best_state, best_ep, since = 1e9, None, -1, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total, seen = 0.0, 0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pb = p_tr[idx].to(dev, non_blocking=True)
            sb = s_tr[idx].to(dev, non_blocking=True)
            yb = y_tr[idx].to(dev, non_blocking=True)
            loss = torch.nn.functional.mse_loss(forward(pb, sb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(y_va), 65536):
                chunks.append(forward(p_va[i:i + 65536], s_va[i:i + 65536]))
            pv = torch.cat(chunks)
            val_mse = float(torch.nn.functional.mse_loss(pv, y_va))
            corr = float(torch.corrcoef(torch.stack([pv, y_va]))[0, 1])
        marker = ""
        if val_mse < best_val:
            best_val, best_ep, since = val_mse, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = " *"
        else:
            since += 1
        print(f"epoch {ep:2d}  train {total / seen:.5f}  val {val_mse:.5f}  "
              f"corr {corr:.4f}  ({time.time() - t0:.0f}s){marker}",
              flush=True)
        if since >= patience:
            print(f"early stop (no val improvement for {patience} epochs)",
                  flush=True)
            break

    model.load_state_dict(best_state)
    clean = strip_dropout(model.cpu(), hidden)
    save_model(clean, hidden, out_path)
    improvement = 1.0 - best_val / base["linear_scalars"]
    print(f"best val MSE {best_val:.5f} (epoch {best_ep})  vs linear-scalars "
          f"{base['linear_scalars']:.5f}  ({improvement:+.1%})  -> {out_path}",
          flush=True)
    return best_val, base


def main():
    ap = argparse.ArgumentParser("value leaf training")
    ap.add_argument("--data", default="results/v4_value/train")
    ap.add_argument("--out", default="results/v4_value/value.pt")
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
    train(args.data.split(","), args.out, hidden, args.epochs, args.batch,
          args.lr, args.weight_decay, args.dropout, args.patience, args.seed)


if __name__ == "__main__":
    main()
