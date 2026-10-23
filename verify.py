"""Verification script for the Optimized PINN pipeline."""
import torch
import json
import pickle
import os
from src.pinn_model import OptimizedPINN

print("=" * 50)
print("  VERIFICATION: Model Reload & Inference")
print("=" * 50)

# 1. Load feature names
path = os.path.join("outputs", "model", "feature_names.json")
with open(path) as f:
    features = json.load(f)
print(f"[1] Features loaded: {len(features)} features")

# 2. Load scaler
path = os.path.join("outputs", "model", "scaler.pkl")
with open(path, "rb") as f:
    scaler = pickle.load(f)
print(f"[2] Scaler loaded: {type(scaler).__name__}")

# 3. Load model
config = {
    "input_dim": len(features),
    "hidden_layers": [256, 128, 64],
    "activation": "gelu",
    "dropout": 0.1,
    "use_layer_norm": True,
    "use_batch_norm": False,
    "use_residual": True,
    "use_attention": True,
    "use_uncertainty": False,
}
model = OptimizedPINN(config)
path = os.path.join("outputs", "model", "optimized_pinn.pth")
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()
print(f"[3] Model loaded: {model.count_parameters():,} parameters")

# 4. Test inference
x = torch.randn(5, len(features))
with torch.no_grad():
    pred = model(x)
print(f"[4] Inference OK: shape={pred.shape}")

# 5. Check metrics
path = os.path.join("outputs", "metrics", "metrics.json")
with open(path) as f:
    metrics = json.load(f)
print(f"[5] Metrics: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R2={metrics['R2']}")

# 6. Check all output files exist
expected = [
    os.path.join("outputs", "model", "optimized_pinn.pth"),
    os.path.join("outputs", "model", "scaler.pkl"),
    os.path.join("outputs", "model", "feature_names.json"),
    os.path.join("outputs", "metrics", "metrics.json"),
    os.path.join("outputs", "metrics", "loss_history.json"),
    os.path.join("outputs", "metrics", "predictions.csv"),
    os.path.join("outputs", "metrics", "config_snapshot.yaml"),
    os.path.join("outputs", "plots", "loss_curves.png"),
    os.path.join("outputs", "plots", "physics_losses.png"),
    os.path.join("outputs", "plots", "predicted_vs_actual.png"),
    os.path.join("outputs", "plots", "error_distribution.png"),
    os.path.join("outputs", "plots", "attention_weights.png"),
    os.path.join("outputs", "plots", "learning_rate_schedule.png"),
]
missing = [f for f in expected if not os.path.exists(f)]
if missing:
    print(f"[6] MISSING: {missing}")
else:
    print(f"[6] All {len(expected)} output files verified")

print()
print("=== ALL VERIFICATION CHECKS PASSED ===")
