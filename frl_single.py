#!/usr/bin/env python3
"""
frl_single_allnodes.py — One-file demo for preprocessing + federated training (FedAvg)
with an option to select **all clients every round** and print per-client metrics.

Quick install:
  pip install torch pandas numpy scikit-learn matplotlib

Typical Jupyter/Colab usage:
  !python frl_single_allnodes.py preprocess --dataset synthetic --out data/processed --clients 10 --input-dim 32 --window 20 --step 10
  !python frl_single_allnodes.py list-nodes --data data/processed
  !python frl_single_allnodes.py simulate --data data/processed --rounds 3 --all-clients --local-epochs 1 --batch-size 32 --lr 1e-3 --seed 42

Import/call from Python:
  import frl_single_allnodes as frl
  frl.main(["simulate","--data","data/processed","--rounds","3","--all-clients"])
"""
import argparse, os, json, copy, random, csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------
# Models
# -------------------
class SmallPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# -------------------
# Utils
# -------------------
def _ensure_dir(p): os.makedirs(p, exist_ok=True)
def _device(): return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_deterministic(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------
# Preprocessing
# -------------------
def make_synthetic(n_rows=20000, n_features=32, n_classes=3, seed=1337):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = (X[:, :3].sum(axis=1) > 0).astype(np.int64) % n_classes
    ts = np.arange(n_rows)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    df["timestamp"] = ts
    return df

def windowize(df, feature_cols, label_col, window, step):
    X_seq, y_seq = [], []
    for start in range(0, len(df) - window + 1, step):
        sl = df.iloc[start:start+window]
        X_seq.append(sl[feature_cols].values.astype(np.float32))
        y_seq.append(int(sl[label_col].values[-1]))
    X_seq = np.stack(X_seq) if len(X_seq) else np.zeros((0, window, len(feature_cols)), dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64) if len(y_seq) else np.zeros((0,), dtype=np.int64)
    return X_seq, y_seq

def split_clients(X, y, clients=10, seed=1337):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    splits = np.array_split(idx, clients)
    return [(X[cidx], y[cidx]) for cidx in splits]

def cmd_preprocess(args):
    _ensure_dir(args.out)
    if args.dataset == "synthetic":
        df = make_synthetic(n_rows=args.rows, n_features=args.input_dim, n_classes=args.n_classes, seed=args.seed)
        feature_cols = [c for c in df.columns if c.startswith("f")]
    else:
        df = pd.read_csv(args.dataset)
        if args.timestamp_col not in df.columns:
            raise ValueError(f"CSV must contain '{args.timestamp_col}' column.")
        feature_cols = [c for c in df.columns if c not in [args.timestamp_col, "label"]][:args.input_dim]
        if len(feature_cols) < args.input_dim:
            raise ValueError(f"Found only {len(feature_cols)} feature columns, need {args.input_dim}.")
        if "label" not in df.columns:
            df["label"] = (df[feature_cols[:3]].sum(axis=1) > 0).astype(int)
        df = df.sort_values(by=args.timestamp_col).reset_index(drop=True)

    df = df.sort_values(by=args.timestamp_col).reset_index(drop=True)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = windowize(df, feature_cols, "label", args.window, args.step)
    splits = split_clients(X, y, clients=args.clients, seed=args.seed)

    meta = {
        "n_clients": len(splits),
        "input_dim": X.shape[-1] if X.size else args.input_dim,
        "window": X.shape[1] if X.size else args.window,
        "n_classes": int(np.max(y)+1) if y.size else args.n_classes,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for i, (Xi, yi) in enumerate(splits):
        obj = {
            "X": torch.tensor(Xi),
            "y": torch.tensor(yi),
            "input_dim": meta["input_dim"],
            "window": meta["window"],
            "n_classes": meta["n_classes"],
        }
        torch.save(obj, os.path.join(args.out, f"client_{i:03d}.pt"))
    print(f"[preprocess] Saved {len(splits)} client tensors to {args.out}")
    print(f"[preprocess] Meta: {meta}")

# -------------------
# Local train + eval
# -------------------
def _pool_inputs(X):  # mean-pool over time window
    return X.mean(dim=1)

def local_train(model, train_tensor, epochs=2, batch_size=32, lr=1e-3, device=None):
    if device is None: device = _device()
    X, y = train_tensor  # X: [N, W, F], y: [N]
    Xp = _pool_inputs(X)
    ds = TensorDataset(Xp, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = loss.item()
    return model.state_dict(), last_loss

@torch.no_grad()
def evaluate(model, train_tensor, device=None):
    if device is None: device = _device()
    X, y = train_tensor
    Xp = _pool_inputs(X).to(device)
    y = y.to(device)
    model = model.to(device).eval()
    logits = model(Xp)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == y).float().mean().item() if y.numel() else 0.0
    return acc

# -------------------
# Federated simulate (FedAvg) with all-clients support
# -------------------
def fedavg(states):
    new_state = copy.deepcopy(states[0])
    for k in new_state.keys():
        for s in states[1:]:
            new_state[k] += s[k]
        new_state[k] /= len(states)
    return new_state

def load_client_tensor(path):
    return torch.load(path, map_location="cpu")

def cmd_list_nodes(args):
    meta_path = os.path.join(args.data, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {args.data}. Run preprocess first.")
    meta = json.load(open(meta_path))
    client_files = sorted([f for f in os.listdir(args.data) if f.startswith("client_") and f.endswith(".pt")])
    print(f"[list-nodes] Total clients: {len(client_files)} (from meta: {meta.get('n_clients')})")
    for cf in client_files:
        obj = load_client_tensor(os.path.join(args.data, cf))
        n = int(obj["X"].shape[0])
        print(f"  - {cf.replace('.pt','')}: samples={n}, window={int(obj['window'])}, feats={int(obj['input_dim'])}, classes={int(obj['n_classes'])}")

def cmd_simulate(args):
    set_deterministic(args.seed)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    input_dim = meta["input_dim"]
    n_clients = meta["n_clients"]
    n_classes = meta["n_classes"]

    client_files = sorted([os.path.join(args.data, f) for f in os.listdir(args.data) if f.startswith("client_") and f.endswith(".pt")])
    assert len(client_files) == n_clients, f"Found {len(client_files)} client files, expected {n_clients}."

    global_model = SmallPolicyNet(input_dim=input_dim, hidden=args.hidden, n_actions=n_classes)

    # Participation log (1/0 per client) + per-client metrics CSV
    part_log_path = os.path.join(args.data, "participation_log.csv")
    metrics_path = os.path.join(args.data, "client_metrics.csv")
    with open(part_log_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round"] + [os.path.basename(p).replace(".pt","") for p in client_files])
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round","client","samples","train_loss","train_acc"])

    for r in range(1, args.rounds+1):
        if args.all_clients:
            selected = client_files[:]  # ALL nodes
        else:
            selected = random.sample(client_files, min(args.clients, len(client_files)))
        print(f"[round {r}/{args.rounds}] selected nodes ({len(selected)}/{len(client_files)}): {[os.path.basename(s).replace('.pt','') for s in selected]}")

        states = []
        # write participation row
        row = [r] + [1 if cf in selected else 0 for cf in client_files]
        with open(part_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # train clients
        for cf in selected:
            ct = load_client_tensor(cf)
            client_model = copy.deepcopy(global_model)
            st, last_loss = local_train(
                client_model,
                (ct["X"], ct["y"]),
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=_device()
            )
            # Evaluate client model on its local data after training
            client_model.load_state_dict(st)
            acc = evaluate(client_model, (ct["X"], ct["y"]), device=_device())
            states.append(st)

            # log per-client metrics
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([r, os.path.basename(cf).replace(".pt",""), int(ct["X"].shape[0]), f"{last_loss:.6f}", f"{acc:.4f}"])

            print(f"   • {os.path.basename(cf).replace('.pt','')}: samples={int(ct['X'].shape[0])}, loss={last_loss:.6f}, acc={acc:.4f}")

        # aggregate
        global_model.load_state_dict(fedavg(states))
        print(f"[round {r}] FedAvg updated.")

    save_path = os.path.join(args.data, "global_model.pt")
    torch.save(global_model.state_dict(), save_path)
    print(f"[simulate] Saved global model to {save_path}")
    print(f"[simulate] Participation -> {part_log_path}")
    print(f"[simulate] Client metrics  -> {metrics_path}")

def cmd_visualize(args):
    import matplotlib.pyplot as plt
    df = pd.read_csv(args.log)
    rounds = df["round"].values
    active = df.drop(columns=["round"]).sum(axis=1).values
    plt.figure()
    plt.plot(rounds, active, marker="o")
    plt.title("Active clients per round")
    plt.xlabel("Round")
    plt.ylabel("Active clients")
    plt.grid(True)
    out = args.out if args.out else os.path.splitext(args.log)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out)
    print(f"[visualize] Saved chart to {out}")

# -------------------
# CLI
# -------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="One-file FRL demo with ALL-CLIENTS option + per-client metrics.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Prepare per-client tensors from a CSV or synthetic data.")
    sp.add_argument("--dataset", required=True, help="'synthetic' or path to CSV")
    sp.add_argument("--out", required=True, help="Output directory for processed client tensors")
    sp.add_argument("--clients", type=int, default=10)
    sp.add_argument("--input-dim", type=int, default=32)
    sp.add_argument("--window", type=int, default=20)
    sp.add_argument("--step", type=int, default=10)
    sp.add_argument("--timestamp-col", type=str, default="timestamp")
    sp.add_argument("--n-classes", type=int, default=3)
    sp.add_argument("--rows", type=int, default=20000, help="Rows for synthetic mode")
    sp.add_argument("--seed", type=int, default=1337)
    sp.set_defaults(func=cmd_preprocess)

    sp2 = sub.add_parser("list-nodes", help="List client nodes and sample counts.")
    sp2.add_argument("--data", required=True, help="Processed data directory (from preprocess).")
    sp2.set_defaults(func=cmd_list_nodes)

    sp3 = sub.add_parser("simulate", help="Run FedAvg rounds with local training.")
    sp3.add_argument("--data", required=True)
    sp3.add_argument("--rounds", type=int, default=5)
    sp3.add_argument("--clients", type=int, default=10, help="Number of clients sampled per round")
    sp3.add_argument("--all-clients", action="store_true", help="Select ALL clients every round")
    sp3.add_argument("--local-epochs", type=int, default=1)
    sp3.add_argument("--batch-size", type=int, default=32)
    sp3.add_argument("--lr", type=float, default=1e-3)
    sp3.add_argument("--hidden", type=int, default=128)
    sp3.add_argument("--seed", type=int, default=1337, help="Deterministic seeds for reproducibility")
    sp3.set_defaults(func=cmd_simulate)

    sp4 = sub.add_parser("visualize", help="Plot active clients per round from participation_log.csv.")
    sp4.add_argument("--log", required=True, help="Path to participation_log.csv created by simulate.")
    sp4.add_argument("--out", default=None, help="Output PNG path.")
    sp4.set_defaults(func=cmd_visualize)
    return ap

def main(argv=None):
    ap = build_argparser()
    args = ap.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
