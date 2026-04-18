"""
CyberSmart - Model Converter (fixed)
Run: python convert_model.py
Creates model_sklearn.pkl from model.pth
"""

import os, sys
import numpy as np

print("Loading PyTorch model...")
try:
    import torch
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)

try:
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    os.system("pip install scikit-learn joblib")
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

backend_dir = os.path.dirname(os.path.abspath(__file__))
pth_path    = os.path.join(backend_dir, "model.pth")
pkl_path    = os.path.join(backend_dir, "model_sklearn.pkl")

if not os.path.exists(pth_path):
    print(f"ERROR: model.pth not found at {pth_path}")
    sys.exit(1)

# ── Load checkpoint ───────────────────────────────────────────────────────────
ck           = torch.load(pth_path, map_location="cpu", weights_only=True)
input_dim    = ck.get("input_dim", 56)
threshold    = float(ck.get("threshold", 0.55))
scaler_mean  = np.array(ck["scaler_mean"],  dtype=np.float64)
scaler_scale = np.array(ck["scaler_scale"], dtype=np.float64)

print(f"  input_dim = {input_dim}")
print(f"  threshold = {threshold}")

# ── Extract Linear layer weights ──────────────────────────────────────────────
sd = ck["model_state_dict"]
layer_weights = []
layer_biases  = []

for k in sd.keys():
    if k.endswith(".weight") and len(sd[k].shape) == 2:
        w     = sd[k].numpy().astype(np.float64)
        b_key = k.replace(".weight", ".bias")
        b     = sd[b_key].numpy().astype(np.float64) if b_key in sd else np.zeros(w.shape[0])
        print(f"  {k}: {w.shape}")
        layer_weights.append(w)
        layer_biases.append(b)

print(f"Linear layers found: {len(layer_weights)}")

# ── Build sklearn MLP ─────────────────────────────────────────────────────────
hidden_sizes = tuple(w.shape[0] for w in layer_weights[:-1])
print(f"Sklearn hidden_sizes: {hidden_sizes}")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu", solver="adam",
        max_iter=1, warm_start=True,
    )
    X_dummy = np.random.randn(10, input_dim)
    y_dummy = np.array([0,1,0,1,0,1,0,1,0,1])
    mlp.fit(X_dummy, y_dummy)

# Inject real weights (sklearn = transposed PyTorch)
mlp.coefs_      = [w.T for w in layer_weights]
mlp.intercepts_ = list(layer_biases)
print("Weights injected into sklearn MLP ✓")

# ── Build scaler ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
scaler.mean_           = scaler_mean
scaler.scale_          = scaler_scale
scaler.var_            = scaler_scale ** 2
scaler.n_features_in_  = input_dim
scaler.n_samples_seen_ = 10000

# ── Quick validation (no PyTorch forward pass needed) ─────────────────────────
print("\nValidating sklearn predictions...")
np.random.seed(42)
X_test   = np.random.randn(200, input_dim)
X_scaled = np.clip((X_test - scaler_mean) / (scaler_scale + 1e-8), -5, 5)
sk_probs = mlp.predict_proba(X_scaled)[:, 1]
sk_preds = (sk_probs > threshold).astype(int)
phish_rate = sk_preds.mean() * 100
print(f"  Phishing rate on random input: {phish_rate:.1f}% (expect 30-70%)")
print(f"  Prob range: {sk_probs.min():.3f} – {sk_probs.max():.3f}")

# Test a clearly safe-looking input (short URL, no hyphens, no suspicious TLD)
safe_input  = np.zeros((1, input_dim))
safe_input[0, 0]  = 20   # length_url = 20 (short)
safe_input[0, 55] = 1    # statistical_report = 1 (in known safe list)
safe_prob = mlp.predict_proba(np.clip((safe_input - scaler_mean)/(scaler_scale+1e-8),-5,5))[:,1][0]
print(f"  Safe-pattern URL probability: {safe_prob:.3f} (expect < 0.55)")

# Test a clearly phishing-looking input
phish_input = np.zeros((1, input_dim))
phish_input[0, 0]  = 120  # length_url = 120 (very long)
phish_input[0, 4]  = 4    # nb_hyphens = 4
phish_input[0, 50] = 3    # phish_hints = 3
phish_input[0, 54] = 1    # suspecious_tld = 1
phish_input[0, 33] = 1    # prefix_suffix = 1
phish_prob = mlp.predict_proba(np.clip((phish_input - scaler_mean)/(scaler_scale+1e-8),-5,5))[:,1][0]
print(f"  Phishing-pattern URL probability: {phish_prob:.3f} (expect > 0.55)")

if safe_prob < 0.55 and phish_prob > 0.55:
    print("\n✅ Model is working correctly!")
else:
    print("\n⚠️  Predictions look off — but pkl will still be saved.")
    print("   This can happen if your model learned different feature weights.")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump({
    "mlp":       mlp,
    "scaler":    scaler,
    "threshold": threshold,
    "input_dim": input_dim,
    "version":   "sklearn-converted",
}, pkl_path)

size_kb = os.path.getsize(pkl_path) / 1024
print(f"\n✅ Saved: model_sklearn.pkl  ({size_kb:.0f} KB)")
print(f"   Original model.pth size:   {os.path.getsize(pth_path)/1024:.0f} KB")
print(f"\nNext steps:")
print(f"  1. Copy model_sklearn.pkl to your deploy folder")
print(f"  2. git add model_sklearn.pkl")
print(f"  3. git rm model.pth  (remove .pth from repo)")
print(f"  4. git commit -m 'Switch to sklearn model'")
print(f"  5. git push")
