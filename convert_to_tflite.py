"""
convert_to_tflite.py  (FINAL FIX)
===================================
Sirf ek cheez fix ki: keras_model.save() → keras_model.export()
Keras 3.x mein SavedModel ke liye .export() use hota hai.

Run:
  cd D:\cybersmart_production\backend
  python convert_to_tflite.py
"""

import os, sys, json, shutil
import numpy as np

print("=" * 55)
print("  CyberSmart — PyTorch to TFLite (Final Fix)")
print("=" * 55)

try:
    import torch
    import torch.nn as nn
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("pip install torch"); sys.exit(1)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError:
    print("pip install tensorflow"); sys.exit(1)

# ── Load checkpoint ────────────────────────────────────────
print("\n[1/5] model.pth load kar raha hai...")
ckpt = torch.load("model.pth", map_location="cpu", weights_only=False)

input_dim    = ckpt.get("input_dim", ckpt.get("n_features", 56))
threshold    = ckpt.get("threshold", 0.55)
scaler_mean  = ckpt.get("scaler_mean",  None)
scaler_scale = ckpt.get("scaler_scale", None)
feat_names   = ckpt.get("feature_names", [f"f{i}" for i in range(input_dim)])
sd           = ckpt["model_state_dict"]

print(f"  input_dim = {input_dim}, threshold = {threshold}")

# ── Detect architecture ────────────────────────────────────
print("\n[2/5] Architecture detect kar raha hai...")

linear_layers = {}
bn_layers = {}
for k, v in sd.items():
    parts = k.split('.')
    if len(parts) >= 3:
        idx   = int(parts[1])
        param = parts[2]
        if param == 'weight' and len(v.shape) == 2:
            linear_layers[idx] = v
        elif param == 'running_mean':
            bn_layers[idx] = True

linear_indices = sorted(linear_layers.keys())
layer_cfg = []
for lin_idx in linear_indices:
    w_tensor = linear_layers[lin_idx]
    units    = w_tensor.shape[0]
    bn_idx   = lin_idx + 1
    has_bn   = bn_idx in bn_layers
    layer_cfg.append((lin_idx, bn_idx if has_bn else None, units))
    print(f"  net.{lin_idx} -> {w_tensor.shape[1]}x{units}"
          f"  BN={'net.'+str(bn_idx) if has_bn else 'None'}")

# ── Save model_info.json ───────────────────────────────────
print("\n[3/5] model_info.json save kar raha hai...")
model_info = {
    "input_dim":     input_dim,
    "threshold":     threshold,
    "feature_names": feat_names,
    "scaler_mean":   scaler_mean,
    "scaler_scale":  scaler_scale,
}
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("  ✓ model_info.json saved")

# ── Build Keras model ──────────────────────────────────────
print("\n[4/5] Keras model bana raha hai...")

def get_w(key):
    return sd[key].numpy()

inp = tf.keras.Input(shape=(input_dim,), name="features")
x = inp

for i, (lin_idx, bn_idx, units) in enumerate(layer_cfg):
    W = get_w(f"net.{lin_idx}.weight").T
    b = get_w(f"net.{lin_idx}.bias")

    x = tf.keras.layers.Dense(
        units,
        kernel_initializer=tf.constant_initializer(W),
        bias_initializer=tf.constant_initializer(b),
        use_bias=True,
        name=f"dense_{i}"
    )(x)

    if bn_idx is not None:
        gamma = get_w(f"net.{bn_idx}.weight")
        beta  = get_w(f"net.{bn_idx}.bias")
        rm    = get_w(f"net.{bn_idx}.running_mean")
        rv    = get_w(f"net.{bn_idx}.running_var")
        x = tf.keras.layers.BatchNormalization(
            gamma_initializer=tf.constant_initializer(gamma),
            beta_initializer=tf.constant_initializer(beta),
            moving_mean_initializer=tf.constant_initializer(rm),
            moving_variance_initializer=tf.constant_initializer(rv),
            name=f"bn_{i}"
        )(x, training=False)

    is_last = (i == len(layer_cfg) - 1)
    if not is_last:
        x = tf.keras.layers.ReLU(name=f"relu_{i}")(x)
    else:
        x = tf.keras.layers.Activation("sigmoid", name="sigmoid_out")(x)

keras_model = tf.keras.Model(inputs=inp, outputs=x, name="PhishingNet")

# ── Verify outputs ─────────────────────────────────────────
test_input = np.random.randn(5, input_dim).astype(np.float32)
tf_out = keras_model.predict(test_input, verbose=0).flatten()
print(f"  Keras outputs: {tf_out}")
print(f"  ✓ Keras model ready")

# ── Convert to TFLite ──────────────────────────────────────
print("\n[5/5] TFLite mein convert kar raha hai...")

SAVED_PATH = "model_tf_saved"

# ── FIX: Use .export() for Keras 3.x / TF 2.11+ ──────────
# Old:  keras_model.save(SAVED_PATH)          ← fails in TF 2.11+
# New:  keras_model.export(SAVED_PATH)        ← correct for SavedModel
if os.path.exists(SAVED_PATH):
    shutil.rmtree(SAVED_PATH)

try:
    # Keras 3.x method
    keras_model.export(SAVED_PATH)
    print(f"  ✓ SavedModel exported via .export()")
except AttributeError:
    # Older Keras fallback
    tf.saved_model.save(keras_model, SAVED_PATH)
    print(f"  ✓ SavedModel exported via tf.saved_model.save()")

# ── TFLite conversion ─────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

if scaler_mean and scaler_scale:
    mean_arr  = np.array(scaler_mean,  dtype=np.float32)
    scale_arr = np.array(scaler_scale, dtype=np.float32)
    def representative_gen():
        for _ in range(300):
            s = np.random.randn(input_dim).astype(np.float32)
            s = np.clip(s * scale_arr + mean_arr, -5, 5)[np.newaxis, :]
            yield [s]
    converter.representative_dataset = representative_gen

tflite_bytes = converter.convert()

TFLITE_PATH = "model.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_bytes)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"  ✓ model.tflite saved ({size_kb:.1f} KB)")

# ── Verify TFLite ─────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"  Input:  shape={inp_det['shape']}  dtype={inp_det['dtype'].__name__}")
print(f"  Output: shape={out_det['shape']}  dtype={out_det['dtype'].__name__}")

# Test inference
tflite_outs = []
for row in test_input:
    data = row[np.newaxis, :].astype(inp_det['dtype'])
    interp.set_tensor(inp_det['index'], data)
    interp.invoke()
    tflite_outs.append(float(interp.get_tensor(out_det['index'])[0][0]))

print(f"  Keras   outputs: {tf_out.tolist()}")
print(f"  TFLite  outputs: {tflite_outs}")
max_diff = max(abs(a - b) for a, b in zip(tf_out, tflite_outs))
print(f"  Max diff: {max_diff:.6f}  {'✓ GOOD' if max_diff < 0.05 else '⚠ CHECK'}")

# Cleanup temp folder
if os.path.exists(SAVED_PATH):
    shutil.rmtree(SAVED_PATH)

# Update model_info with size
model_info["model_size_kb"] = round(size_kb, 1)
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"""
{"=" * 55}
  CONVERSION COMPLETE! ✓
{"=" * 55}

  Bane hue files (backend folder mein):
    model.tflite      ({size_kb:.0f} KB)
    model_info.json

  AB YEH KARO:

  Step 1 — assets folder banao:
    mkdir D:\\cybersmart_flutter\\assets

  Step 2 — Files copy karo:
    copy model.tflite D:\\cybersmart_flutter\\assets\\
    copy model_info.json D:\\cybersmart_flutter\\assets\\

  Step 3 — Flutter run:
    cd D:\\cybersmart_flutter
    flutter pub get
    flutter run
""")
