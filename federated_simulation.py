"""
CyberSmart — Federated Learning Simulation (Standalone, v2)
============================================================
Run:  cd D:\cybersmart_production\federated_learning
      python federated_simulation.py

Does NOT import from ../backend/model.py — fully self-contained.
Loads dataset from ../backend/dataset_phishing.csv
"""

import os, sys, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
N_CLIENTS    = 5
N_ROUNDS     = 10
LOCAL_EPOCHS = 3
FRACTION     = 0.8        # fraction of clients selected per round
BATCH_SIZE   = 64
LR           = 0.001
INPUT_DIM    = 56
HIDDEN       = (128, 64, 32, 16)
THRESHOLD    = 0.55

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "backend", "dataset_phishing.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL  (56-feature MLP — matches production model exactly)
# ══════════════════════════════════════════════════════════════════════════════
class PhishNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    try:
        import pandas as pd
    except ImportError:
        os.system("pip install pandas --break-system-packages")
        import pandas as pd

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        print("        Using synthetic data instead...")
        X = np.random.randn(11430, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, 2, 11430).astype(np.float32)
        return X, y, INPUT_DIM, 11430

    df = pd.read_csv(DATASET_PATH)

    # Drop non-feature columns
    drop_cols = [c for c in ["url", "status", "id"] if c in df.columns]
    label_col = "status" if "status" in df.columns else df.columns[-1]

    y_raw = df[label_col].values
    X_df  = df.drop(columns=drop_cols)

    # Encode label — handles 'phishing'/'legitimate' strings OR numeric 0/1
    try:
        y = y_raw.astype(np.float32)
    except (ValueError, TypeError):
        y = (np.array(y_raw) == "phishing").astype(np.float32)

    X = X_df.select_dtypes(include=[np.number]).values.astype(np.float32)
    print(f"[Data] Loaded {len(X)} samples, {X.shape[1]} features")
    print(f"[Data] Phishing: {int(y.sum())} ({y.mean()*100:.1f}%)  "
          f"Legitimate: {int((1-y).sum())} ({(1-y.mean())*100:.1f}%)")
    return X, y, X.shape[1], len(X)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA PARTITIONING  (IID split across clients)
# ══════════════════════════════════════════════════════════════════════════════
def partition_data(X, y, n_clients):
    idx = np.random.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    partitions = []
    for i, s in enumerate(splits):
        Xi, yi = X[s], y[s]
        phish_rate = yi.mean() * 100
        print(f"   Client {i}: {len(Xi):>4d} samples  |  phishing {phish_rate:.1f}%")
        partitions.append((Xi, yi))
    return partitions

# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLIENT
# ══════════════════════════════════════════════════════════════════════════════
class FederatedClient:
    def __init__(self, client_id, X, y, scaler):
        self.id      = client_id
        self.scaler  = scaler
        X_sc         = scaler.transform(X)
        X_sc         = np.clip(X_sc, -5, 5).astype(np.float32)
        # Train/val split 80/20
        n            = len(X_sc)
        split        = int(0.8 * n)
        self.X_train = torch.tensor(X_sc[:split])
        self.y_train = torch.tensor(y[:split])
        self.X_val   = torch.tensor(X_sc[split:])
        self.y_val   = torch.tensor(y[split:])
        self.model   = PhishNet(input_dim=X_sc.shape[1])
        self.n_train = split

    def set_weights(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def local_train(self, epochs=LOCAL_EPOCHS):
        self.model.train()
        opt      = torch.optim.Adam(self.model.parameters(), lr=LR)
        criterion= nn.BCELoss()
        ds       = TensorDataset(self.X_train, self.y_train.unsqueeze(1))
        loader   = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        total_loss = 0.0
        for _ in range(epochs):
            for bX, by in loader:
                opt.zero_grad()
                loss = criterion(self.model(bX), by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
        # Validation
        self.model.eval()
        with torch.no_grad():
            probs = self.model(self.X_val).numpy().flatten()
            preds = (probs > THRESHOLD).astype(int)
            acc   = accuracy_score(self.y_val.numpy(), preds)
            f1    = f1_score(self.y_val.numpy(), preds, zero_division=0)
            auc   = roc_auc_score(self.y_val.numpy(), probs)
        return {
            "client_id": self.id,
            "n_samples":  self.n_train,
            "val_acc":    acc,
            "val_f1":     f1,
            "val_auc":    auc,
            "avg_loss":   total_loss / max(len(loader)*epochs, 1),
        }

# ══════════════════════════════════════════════════════════════════════════════
# 5.  SERVER  (FedAvg aggregation)
# ══════════════════════════════════════════════════════════════════════════════
class FederatedServer:
    def __init__(self, clients, X_test, y_test):
        self.clients   = clients
        self.X_test    = X_test
        self.y_test    = y_test
        self.global_model = PhishNet(input_dim=X_test.shape[1])
        # Push initial weights to all clients
        for c in clients:
            c.set_weights(self.global_model.state_dict())
        print(f"[Server] Initialized with {len(clients)} clients")

    def fed_avg(self, selected_clients):
        """Weighted FedAvg — weight by number of local samples."""
        total_samples = sum(c.n_train for c in selected_clients)
        new_state     = copy.deepcopy(selected_clients[0].get_weights())
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
        for c in selected_clients:
            w = c.n_train / total_samples
            for key in new_state:
                new_state[key] += w * c.get_weights()[key].float()
        self.global_model.load_state_dict(new_state)
        # Broadcast to ALL clients
        for c in self.clients:
            c.set_weights(new_state)

    def evaluate_global(self):
        self.global_model.eval()
        with torch.no_grad():
            probs = self.global_model(self.X_test).numpy().flatten()
            preds = (probs > THRESHOLD).astype(int)
            y_np  = self.y_test.numpy()
            acc   = accuracy_score(y_np, preds)
            f1    = f1_score(y_np, preds, zero_division=0)
            auc   = roc_auc_score(y_np, probs)
            cm    = confusion_matrix(y_np, preds)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
        return {
            "accuracy":  acc,
            "f1":        f1,
            "auc":       auc,
            "precision": precision,
            "recall":    recall,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    def run_federation(self, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
                       fraction=FRACTION):
        history = []
        k = max(1, int(len(self.clients) * fraction))

        sep = "=" * 55
        print(sep)
        print(f"  CyberSmart — Federated Learning Simulation")
        print(f"  Clients: {len(self.clients)}  |  Rounds: {n_rounds}")
        print(f"  Local epochs: {local_epochs}  |  Fraction: {fraction*100:.0f}%")
        print(sep)

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            selected = random.sample(self.clients, k)
            sel_ids  = [c.id for c in selected]
            print(f"\n📡 Round {rnd}/{n_rounds}")
            print(f"   Selected clients: {sel_ids}")

            round_results = []
            for c in selected:
                res = c.local_train(epochs=local_epochs)
                round_results.append(res)
                print(f"   Client {c.id}: acc={res['val_acc']:.4f}  "
                      f"f1={res['val_f1']:.4f}  auc={res['val_auc']:.4f}  "
                      f"loss={res['avg_loss']:.4f}")

            self.fed_avg(selected)
            global_res = self.evaluate_global()
            elapsed    = time.time() - t0

            print(f"   ─────────────────────────────────────────────")
            print(f"   🌐 Global  acc={global_res['accuracy']:.4f}  "
                  f"f1={global_res['f1']:.4f}  "
                  f"auc={global_res['auc']:.4f}  "
                  f"({elapsed:.1f}s)")

            history.append({
                "round":       rnd,
                "global":      global_res,
                "clients":     round_results,
                "elapsed_s":   elapsed,
            })

        return history

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_start = time.time()

    # --- Load & scale data ---
    X, y, feat_dim, n_samples = load_data()
    scaler = StandardScaler()

    # Hold out 15% as global test set
    idx   = np.random.permutation(n_samples)
    n_test= int(0.15 * n_samples)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_test_raw = X[test_idx]; y_test = y[test_idx]
    X_train    = X[train_idx]; y_train= y[train_idx]

    scaler.fit(X_train)
    X_test_sc  = np.clip(scaler.transform(X_test_raw), -5, 5).astype(np.float32)
    X_test_t   = torch.tensor(X_test_sc)
    y_test_t   = torch.tensor(y_test)

    # --- Partition among clients ---
    print(f"\n[Data] Partitioning {len(X_train)} training samples → {N_CLIENTS} clients:")
    partitions = partition_data(X_train, y_train, N_CLIENTS)

    # --- Build clients ---
    clients = [FederatedClient(i, Xi, yi, scaler)
               for i, (Xi, yi) in enumerate(partitions)]

    # --- Centralised baseline (single model, same data, 10 epochs) ---
    print("\n[Baseline] Training centralised model for comparison...")
    cent_model = PhishNet(input_dim=feat_dim)
    cent_opt   = torch.optim.Adam(cent_model.parameters(), lr=LR)
    cent_crit  = nn.BCELoss()
    X_tr_sc    = np.clip(scaler.transform(X_train), -5, 5).astype(np.float32)
    X_tr_t     = torch.tensor(X_tr_sc)
    y_tr_t     = torch.tensor(y_train)
    ds_cent    = TensorDataset(X_tr_t, y_tr_t.unsqueeze(1))
    loader_c   = DataLoader(ds_cent, batch_size=BATCH_SIZE, shuffle=True)
    cent_model.train()
    for ep in range(N_ROUNDS * LOCAL_EPOCHS):   # same total compute budget
        for bX, by in loader_c:
            cent_opt.zero_grad()
            cent_crit(cent_model(bX), by).backward()
            cent_opt.step()
    cent_model.eval()
    with torch.no_grad():
        c_probs = cent_model(X_test_t).numpy().flatten()
        c_preds = (c_probs > THRESHOLD).astype(int)
        cent_acc = accuracy_score(y_test_t.numpy(), c_preds)
        cent_f1  = f1_score(y_test_t.numpy(), c_preds, zero_division=0)
        cent_auc = roc_auc_score(y_test_t.numpy(), c_probs)
    print(f"   Centralised baseline → acc={cent_acc:.4f}  f1={cent_f1:.4f}  auc={cent_auc:.4f}")

    # --- Run Federation ---
    server  = FederatedServer(clients, X_test_t, y_test_t)
    history = server.run_federation(n_rounds=N_ROUNDS,
                                    local_epochs=LOCAL_EPOCHS,
                                    fraction=FRACTION)

    # ── Final Summary ─────────────────────────────────────────────────────────
    final    = history[-1]["global"]
    total_t  = time.time() - total_start

    print("\n" + "=" * 55)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Dataset          : {n_samples} URLs  ({feat_dim} features)")
    print(f"  Clients          : {N_CLIENTS}")
    print(f"  Rounds           : {N_ROUNDS}")
    print(f"  Local Epochs/Rnd : {LOCAL_EPOCHS}")
    print(f"  Client Fraction  : {int(FRACTION*100)}%")
    print(f"  Total FL time    : {total_t:.1f}s")
    print("-" * 55)
    print(f"  FEDERATED MODEL:")
    print(f"    Accuracy       : {final['accuracy']*100:.2f}%")
    print(f"    F1 Score       : {final['f1']:.4f}")
    print(f"    AUC-ROC        : {final['auc']:.4f}")
    print(f"    Precision      : {final['precision']:.4f}")
    print(f"    Recall         : {final['recall']:.4f}")
    print(f"    TP={final['tp']}  TN={final['tn']}  FP={final['fp']}  FN={final['fn']}")
    print("-" * 55)
    print(f"  CENTRALISED BASELINE:")
    print(f"    Accuracy       : {cent_acc*100:.2f}%")
    print(f"    F1 Score       : {cent_f1:.4f}")
    print(f"    AUC-ROC        : {cent_auc:.4f}")
    print("-" * 55)
    acc_gap = (final['accuracy'] - cent_acc) * 100
    print(f"  FL vs Centralised gap : {acc_gap:+.2f}%")
    print("=" * 55)

    # Round-by-round accuracy table
    print("\n  Round-by-Round Global Accuracy:")
    print("  " + "-"*35)
    print(f"  {'Round':>6}  {'Accuracy':>9}  {'F1':>7}  {'AUC':>7}")
    print("  " + "-"*35)
    for h in history:
        g = h["global"]
        print(f"  {h['round']:>6}  {g['accuracy']*100:>8.2f}%  "
              f"{g['f1']:>7.4f}  {g['auc']:>7.4f}")
    print("  " + "-"*35)
    print("\n✅ Simulation complete. Copy the output above for your research paper.")
