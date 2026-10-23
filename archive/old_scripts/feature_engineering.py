"""
feature_engineering.py
----------------------
Phase 4: Feature engineering and selection for earthquake magnitude prediction.

Performs:
    1. Correlation analysis & multicollinearity detection
    2. Mutual Information analysis
    3. XGBoost feature importance
    4. Recursive Feature Elimination (RFE)
    5. Feature variance analysis
    6. Remove degenerate and redundant features
    7. Generate physics-inspired derived features
    8. Save optimized feature set

All results saved to outputs/plots/feature_engineering/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Paths
DATASET_PATH = "earthquake_29_features.csv"
FE_DIR = os.path.join("outputs", "plots", "feature_engineering")
REPORT_PATH = os.path.join("outputs", "metrics", "feature_engineering_report.json")

# Features to always remove (degenerate)
DEGENERATE_FEATURES = ["dominant_freq", "predominant_period", "zero_crossings"]

# Highly correlated feature groups (keep first, remove rest)
# Determined from Phase 3 analysis
REDUNDANT_GROUPS = [
    # pga, max_val, envelope_peak are identical
    {"keep": "pga", "remove": ["max_val", "envelope_peak"]},
    # mean, rms, abs_mean, cav are near-identical
    {"keep": "rms", "remove": ["mean", "abs_mean", "cav"]},
    # p90, p95 are highly correlated with pga
    {"keep": "p75", "remove": ["p90", "p95"]},
    # energy, arias, spectral_energy are near-identical
    {"keep": "energy", "remove": ["arias", "spectral_energy"]},
]


def setup():
    """Create output directory."""
    os.makedirs(FE_DIR, exist_ok=True)
    print(f"[Feature Engineering] Output directory: {FE_DIR}")


def load_data():
    """Load dataset."""
    df = pd.read_csv(DATASET_PATH)
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])
    return df


def step1_remove_degenerate(df):
    """Remove degenerate features with zero or near-zero variance."""
    print("\n" + "=" * 60)
    print("  STEP 1: REMOVE DEGENERATE FEATURES")
    print("=" * 60)

    removed = []
    for feat in DEGENERATE_FEATURES:
        if feat in df.columns:
            std = df[feat].std()
            zero_pct = (df[feat] == 0).sum() / len(df) * 100
            print(f"  Removing '{feat}': std={std:.6f}, zeros={zero_pct:.1f}%")
            df = df.drop(columns=[feat])
            removed.append(feat)

    print(f"  Removed {len(removed)} degenerate features: {removed}")
    print(f"  Remaining features: {len(df.columns) - 1}")
    return df, removed


def step2_remove_redundant(df):
    """Remove highly redundant features based on correlation analysis."""
    print("\n" + "=" * 60)
    print("  STEP 2: REMOVE REDUNDANT FEATURES")
    print("=" * 60)

    removed = []
    features = [c for c in df.columns if c != "magnitude"]
    corr = df[features].corr()

    for group in REDUNDANT_GROUPS:
        keep = group["keep"]
        for feat in group["remove"]:
            if feat in df.columns and keep in df.columns:
                r = corr.loc[keep, feat] if feat in corr.columns else 0
                print(f"  Removing '{feat}' (corr with '{keep}': r={r:.4f})")
                df = df.drop(columns=[feat])
                removed.append(feat)

    print(f"\n  Removed {len(removed)} redundant features: {removed}")
    print(f"  Remaining features: {len(df.columns) - 1}")
    return df, removed


def step3_correlation_analysis(df):
    """Compute and visualize feature-target correlations."""
    print("\n" + "=" * 60)
    print("  STEP 3: CORRELATION ANALYSIS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    valid = [c for c in features if df[c].std() > 1e-10]

    target_corr = df[valid + ["magnitude"]].corr()["magnitude"].drop("magnitude")
    target_corr = target_corr.sort_values(key=abs, ascending=False)

    print(f"\n  {'Feature':<25} {'Correlation':>12}")
    print("  " + "-" * 40)
    for feat, r in target_corr.items():
        print(f"  {feat:<25} {r:>+12.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_corr = target_corr.sort_values()
    colors = ["#D32F2F" if r < 0 else "#4CAF50" for r in sorted_corr.values]
    ax.barh(sorted_corr.index, sorted_corr.values, color=colors, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Magnitude", fontsize=13)
    ax.set_title("Feature–Magnitude Correlations (After Cleaning)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FE_DIR, "correlations_cleaned.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return target_corr


def step4_mutual_information(df):
    """Compute mutual information between features and target."""
    print("\n" + "=" * 60)
    print("  STEP 4: MUTUAL INFORMATION ANALYSIS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    X = df[features].values
    y = df["magnitude"].values

    # Handle any NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)

    print(f"\n  {'Feature':<25} {'MI Score':>12}")
    print("  " + "-" * 40)
    for feat, score in mi_series.items():
        print(f"  {feat:<25} {score:>12.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_mi = mi_series.sort_values()
    ax.barh(sorted_mi.index, sorted_mi.values, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Mutual Information Score", fontsize=13)
    ax.set_title("Feature Importance (Mutual Information)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FE_DIR, "mutual_information.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mi_series


def step5_xgboost_importance(df):
    """Compute feature importance using XGBoost (if available)."""
    print("\n" + "=" * 60)
    print("  STEP 5: XGBOOST FEATURE IMPORTANCE")
    print("=" * 60)

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  ⚠️ XGBoost not available. Skipping.")
        return None

    features = [c for c in df.columns if c != "magnitude"]
    X = df[features].values
    y = df["magnitude"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"  XGBoost baseline R² = {r2:.4f}")

    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)

    print(f"\n  {'Feature':<25} {'Importance':>12}")
    print("  " + "-" * 40)
    for feat, imp in importance.items():
        print(f"  {feat:<25} {imp:>12.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_imp = importance.sort_values()
    colors = plt.cm.YlOrRd(sorted_imp.values / sorted_imp.values.max())
    ax.barh(sorted_imp.index, sorted_imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (XGBoost)", fontsize=13)
    ax.set_title(f"XGBoost Feature Importance (R²={r2:.4f})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FE_DIR, "xgboost_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return importance, r2


def step6_add_derived_features(df):
    """Add physics-inspired derived features."""
    print("\n" + "=" * 60)
    print("  STEP 6: ADD PHYSICS-INSPIRED FEATURES")
    print("=" * 60)

    added = []

    # 1. Log spectral centroid: log(f) relates to magnitude
    if "spectral_centroid" in df.columns:
        df["log_spectral_centroid"] = np.log(df["spectral_centroid"].clip(lower=1e-8))
        added.append("log_spectral_centroid")
        print(f"  Added: log_spectral_centroid = log(spectral_centroid)")

    # 2. Energy per duration: energy density
    if "energy" in df.columns and "duration" in df.columns:
        df["energy_per_duration"] = df["energy"] / df["duration"].clip(lower=1e-8)
        added.append("energy_per_duration")
        print(f"  Added: energy_per_duration = energy / duration")

    # 3. Log energy per duration
    if "energy_per_duration" in df.columns:
        df["log_energy_per_duration"] = np.log(df["energy_per_duration"].clip(lower=1e-8))
        added.append("log_energy_per_duration")
        print(f"  Added: log_energy_per_duration")

    # 4. PGA to energy ratio (amplitude vs energy)
    if "pga" in df.columns and "energy" in df.columns:
        df["pga_energy_ratio"] = df["pga"] / (df["energy"].clip(lower=1e-8))
        added.append("pga_energy_ratio")
        print(f"  Added: pga_energy_ratio = pga / energy")

    # 5. Log PGA: log₁₀(PGA) is commonly used in GMPEs
    if "pga" in df.columns:
        df["log_pga"] = np.log10(df["pga"].clip(lower=1e-8))
        added.append("log_pga")
        print(f"  Added: log_pga = log10(pga)")

    # 6. Bandwidth to centroid ratio
    if "bandwidth" in df.columns and "spectral_centroid" in df.columns:
        df["bandwidth_centroid_ratio"] = df["bandwidth"] / df["spectral_centroid"].clip(lower=1e-8)
        added.append("bandwidth_centroid_ratio")
        print(f"  Added: bandwidth_centroid_ratio = bandwidth / spectral_centroid")

    # 7. Kurtosis * skewness interaction (nonlinear shape descriptor)
    if "kurtosis" in df.columns and "skewness" in df.columns:
        df["kurtosis_skewness"] = df["kurtosis"] * df["skewness"]
        added.append("kurtosis_skewness")
        print(f"  Added: kurtosis_skewness = kurtosis × skewness")

    # 8. Range to RMS ratio (crest factor proxy)
    if "range" in df.columns and "rms" in df.columns:
        df["range_rms_ratio"] = df["range"] / df["rms"].clip(lower=1e-8)
        added.append("range_rms_ratio")
        print(f"  Added: range_rms_ratio = range / rms")

    # 9. IQR to std ratio (robustness of spread)
    if "iqr" in df.columns and "std" in df.columns:
        df["iqr_std_ratio"] = df["iqr"] / df["std"].clip(lower=1e-8)
        added.append("iqr_std_ratio")
        print(f"  Added: iqr_std_ratio = iqr / std")

    # 10. Squared log energy (polynomial feature)
    if "log_energy" in df.columns:
        df["log_energy_sq"] = df["log_energy"] ** 2
        added.append("log_energy_sq")
        print(f"  Added: log_energy_sq = log_energy²")

    print(f"\n  Total derived features added: {len(added)}")

    # Check correlations of new features with target
    print(f"\n  New feature correlations with magnitude:")
    for feat in added:
        if df[feat].std() > 1e-10:
            r = df[feat].corr(df["magnitude"])
            print(f"    {feat:<30} r = {r:+.4f}")

    return df, added


def step7_final_feature_selection(df, mi_scores, xgb_importance):
    """Select the final feature set based on all analyses."""
    print("\n" + "=" * 60)
    print("  STEP 7: FINAL FEATURE SELECTION")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]

    # Combine scores: normalize MI and XGBoost importance
    scores = pd.DataFrame(index=features)

    if mi_scores is not None:
        mi_norm = mi_scores.reindex(features).fillna(0)
        if mi_norm.max() > 0:
            mi_norm = mi_norm / mi_norm.max()
        scores["mi"] = mi_norm

    if xgb_importance is not None:
        xgb_norm = xgb_importance.reindex(features).fillna(0)
        if xgb_norm.max() > 0:
            xgb_norm = xgb_norm / xgb_norm.max()
        scores["xgb"] = xgb_norm

    # Correlation with target
    target_corr = df[features + ["magnitude"]].corr()["magnitude"].drop("magnitude").abs()
    corr_norm = target_corr.reindex(features).fillna(0)
    if corr_norm.max() > 0:
        corr_norm = corr_norm / corr_norm.max()
    scores["corr"] = corr_norm

    # Combined score (weighted average)
    weights = {"mi": 0.3, "xgb": 0.4, "corr": 0.3}
    scores["combined"] = sum(scores.get(k, 0) * v for k, v in weights.items()
                             if k in scores.columns)
    scores = scores.sort_values("combined", ascending=False)

    print(f"\n  {'Feature':<30} {'Combined':>10} {'MI':>8} {'XGB':>8} {'Corr':>8}")
    print("  " + "-" * 70)
    for feat in scores.index:
        row = scores.loc[feat]
        print(f"  {feat:<30} {row.get('combined', 0):>10.4f} "
              f"{row.get('mi', 0):>8.4f} {row.get('xgb', 0):>8.4f} "
              f"{row.get('corr', 0):>8.4f}")

    # Keep all features (let the model learn) but document rankings
    # We remove only clearly useless features (already done in steps 1-2)
    selected = features  # keep all cleaned features

    print(f"\n  Final feature count: {len(selected)}")

    return selected, scores


def save_engineered_dataset(df, selected_features, removed_features, added_features):
    """Save the engineered dataset and feature report."""
    print("\n" + "=" * 60)
    print("  SAVING ENGINEERED DATASET")
    print("=" * 60)

    # Save engineered dataset
    output_path = "earthquake_engineered.csv"
    df[selected_features + ["magnitude"]].to_csv(output_path, index=False)
    print(f"  Dataset saved → {output_path}")
    print(f"  Shape: {df[selected_features + ['magnitude']].shape}")

    # Save report
    report = {
        "original_features": 29,
        "removed_degenerate": DEGENERATE_FEATURES,
        "removed_redundant": [f for group in REDUNDANT_GROUPS for f in group["remove"]],
        "added_features": added_features,
        "final_features": selected_features,
        "final_feature_count": len(selected_features),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)
    print(f"  Report saved → {REPORT_PATH}")


def main():
    """Run the complete feature engineering pipeline."""
    print("\n" + "█" * 60)
    print("  PHASE 4: FEATURE ENGINEERING")
    print("█" * 60)

    setup()
    df = load_data()

    # Step 1: Remove degenerate features
    df, removed_degenerate = step1_remove_degenerate(df)

    # Step 2: Remove redundant features
    df, removed_redundant = step2_remove_redundant(df)

    # Step 3: Correlation analysis
    target_corr = step3_correlation_analysis(df)

    # Step 4: Mutual information
    mi_scores = step4_mutual_information(df)

    # Step 5: XGBoost importance
    xgb_result = step5_xgboost_importance(df)
    xgb_importance = xgb_result[0] if xgb_result else None
    xgb_r2 = xgb_result[1] if xgb_result else None

    # Step 6: Add derived features
    df, added_features = step6_add_derived_features(df)

    # Re-run MI and XGBoost with new features
    print("\n  Re-evaluating with derived features...")
    mi_scores_new = step4_mutual_information(df)
    xgb_result_new = step5_xgboost_importance(df)
    xgb_importance_new = xgb_result_new[0] if xgb_result_new else None

    # Step 7: Final selection
    selected, scores = step7_final_feature_selection(
        df, mi_scores_new, xgb_importance_new
    )

    # Save
    all_removed = removed_degenerate + removed_redundant
    save_engineered_dataset(df, selected, all_removed, added_features)

    print("\n" + "█" * 60)
    print("  FEATURE ENGINEERING COMPLETE")
    print(f"  Original: 29 features → Final: {len(selected)} features")
    print(f"  Removed: {len(all_removed)} | Added: {len(added_features)}")
    if xgb_r2 is not None:
        print(f"  XGBoost baseline R² (original features): {xgb_r2:.4f}")
    if xgb_result_new:
        print(f"  XGBoost baseline R² (engineered features): {xgb_result_new[1]:.4f}")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
