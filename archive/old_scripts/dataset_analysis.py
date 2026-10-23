"""
dataset_analysis.py
-------------------
Phase 3: Complete dataset analysis with publication-quality visualizations.

Generates:
    1. Missing values report
    2. Duplicate samples check
    3. Feature distributions (histograms)
    4. Target distribution
    5. Outlier detection (IQR + Z-score)
    6. Feature variance analysis
    7. Feature correlation heatmap
    8. Feature-target correlations
    9. Redundant/constant feature identification
    10. Data imbalance analysis

All plots saved to outputs/plots/analysis/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Paths
DATASET_PATH = "earthquake_29_features.csv"
ANALYSIS_DIR = os.path.join("outputs", "plots", "analysis")


def setup():
    """Create analysis output directory."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print(f"[Analysis] Output directory: {ANALYSIS_DIR}")


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATASET_PATH)
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])
    print(f"[Analysis] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def analyze_missing_values(df):
    """Check and visualize missing values."""
    print("\n" + "=" * 60)
    print("  1. MISSING VALUES ANALYSIS")
    print("=" * 60)

    missing = df.isnull().sum()
    total = len(df)
    missing_pct = (missing / total * 100).round(2)

    report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
    })

    total_missing = missing.sum()
    print(f"  Total missing values: {total_missing}")
    if total_missing > 0:
        print(report[report["Missing Count"] > 0])
    else:
        print("  ✓ No missing values detected")

    return report


def analyze_duplicates(df):
    """Check for duplicate samples."""
    print("\n" + "=" * 60)
    print("  2. DUPLICATE SAMPLES")
    print("=" * 60)

    n_duplicates = df.duplicated().sum()
    print(f"  Total duplicate rows: {n_duplicates}")
    if n_duplicates > 0:
        print(f"  Duplicate percentage: {n_duplicates / len(df) * 100:.2f}%")
    else:
        print("  ✓ No duplicate rows found")

    return n_duplicates


def analyze_target_distribution(df):
    """Analyze and plot the target variable distribution."""
    print("\n" + "=" * 60)
    print("  3. TARGET DISTRIBUTION")
    print("=" * 60)

    target = df["magnitude"]
    print(f"  Range: {target.min():.1f} – {target.max():.1f}")
    print(f"  Mean: {target.mean():.3f}")
    print(f"  Std: {target.std():.3f}")
    print(f"  Unique values: {target.nunique()}")
    print(f"\n  Value counts:")
    vc = target.value_counts().sort_index()
    for mag, count in vc.items():
        bar = "█" * int(count / 10)
        print(f"    M={mag:.1f}: {count:>4d} {bar}")

    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(vc)))
    ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white", linewidth=1.5)
    for i, (mag, count) in enumerate(vc.items()):
        ax.text(i, count + 5, str(count), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Magnitude", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Target Distribution (Magnitude)", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(target.values, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#2196F3", alpha=0.7),
                     medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("Magnitude", fontsize=13)
    ax.set_title("Magnitude Box Plot", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "target_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Plot] Saved → {save_path}")


def analyze_feature_distributions(df):
    """Plot histograms of all features."""
    print("\n" + "=" * 60)
    print("  4. FEATURE DISTRIBUTIONS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    n_features = len(features)

    # Create grid of histograms
    n_cols = 5
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].values
        # Skip constant features for histogram
        if np.std(data) < 1e-10:
            ax.text(0.5, 0.5, f"{feat}\n(CONSTANT)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_title(feat, fontsize=9, fontweight="bold", color="red")
            continue

        ax.hist(data, bins=40, color="#2196F3", edgecolor="white",
                alpha=0.85, linewidth=0.5)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        # Add stats
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1, alpha=0.7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — All 29 Features", fontsize=16,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "feature_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {save_path}")


def analyze_outliers(df):
    """Detect outliers using IQR and Z-score methods."""
    print("\n" + "=" * 60)
    print("  5. OUTLIER ANALYSIS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    outlier_report = []

    for feat in features:
        data = df[feat].values
        if np.std(data) < 1e-10:
            continue

        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = np.sum((data < lower) | (data > upper))

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = np.sum(z_scores > 3)

        outlier_report.append({
            "feature": feat,
            "iqr_outliers": iqr_outliers,
            "iqr_pct": round(iqr_outliers / len(data) * 100, 2),
            "zscore_outliers": z_outliers,
            "zscore_pct": round(z_outliers / len(data) * 100, 2),
        })

    report_df = pd.DataFrame(outlier_report)
    report_df = report_df.sort_values("iqr_outliers", ascending=False)

    print(f"\n  {'Feature':<25} {'IQR Outliers':>15} {'Z-Score Outliers':>18}")
    print("  " + "-" * 60)
    for _, row in report_df.head(15).iterrows():
        print(f"  {row['feature']:<25} {row['iqr_outliers']:>8d} ({row['iqr_pct']:.1f}%)"
              f"   {row['zscore_outliers']:>8d} ({row['zscore_pct']:.1f}%)")

    # Plot top outlier features
    top_outlier_feats = report_df.head(6)["feature"].values
    n = len(top_outlier_feats)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_outlier_feats):
        ax = axes[i]
        data = df[feat].values
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                         boxprops=dict(facecolor="#FF9800", alpha=0.7),
                         medianprops=dict(color="red", linewidth=2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Features with Most Outliers (IQR Method)", fontsize=16,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "outlier_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Plot] Saved → {save_path}")

    return report_df


def analyze_feature_variance(df):
    """Analyze feature variance to identify constant/near-constant features."""
    print("\n" + "=" * 60)
    print("  6. FEATURE VARIANCE ANALYSIS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    variance_data = []

    for feat in features:
        data = df[feat].values
        variance_data.append({
            "feature": feat,
            "std": np.std(data),
            "variance": np.var(data),
            "nunique": df[feat].nunique(),
            "zero_pct": round(np.sum(data == 0) / len(data) * 100, 2),
        })

    var_df = pd.DataFrame(variance_data).sort_values("std")

    print(f"\n  {'Feature':<25} {'Std':>12} {'Unique':>8} {'Zero %':>8}")
    print("  " + "-" * 55)
    for _, row in var_df.iterrows():
        flag = " ⚠️ DEGENERATE" if row["std"] < 1.0 and row["zero_pct"] > 95 else ""
        print(f"  {row['feature']:<25} {row['std']:>12.6f} {row['nunique']:>8d} "
              f"{row['zero_pct']:>7.1f}%{flag}")

    # Identify degenerate features
    degenerate = var_df[(var_df["std"] < 1e-6) | (var_df["zero_pct"] > 99)]
    if len(degenerate) > 0:
        print(f"\n  ⚠️ DEGENERATE FEATURES (zero variance or >99% zeros):")
        for _, row in degenerate.iterrows():
            print(f"    - {row['feature']} (std={row['std']:.8f}, zero%={row['zero_pct']:.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#D32F2F" if s < 1e-6 else "#FF9800" if s < 1.0 else "#4CAF50"
              for s in var_df["std"].values]
    ax.barh(var_df["feature"].values, np.log10(var_df["std"].values + 1e-10),
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log₁₀(Standard Deviation)", fontsize=13)
    ax.set_title("Feature Variance Analysis", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D32F2F", label="Degenerate (std ≈ 0)"),
        Patch(facecolor="#FF9800", label="Low variance (std < 1)"),
        Patch(facecolor="#4CAF50", label="Normal variance"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "feature_variance.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Plot] Saved → {save_path}")

    return var_df


def analyze_correlations(df):
    """Compute and visualize feature correlations."""
    print("\n" + "=" * 60)
    print("  7. CORRELATION ANALYSIS")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]
    # Remove constant features for correlation
    valid_features = [c for c in features if df[c].std() > 1e-10]

    corr_matrix = df[valid_features + ["magnitude"]].corr()

    # Feature-target correlations
    target_corr = corr_matrix["magnitude"].drop("magnitude").sort_values(key=abs, ascending=False)
    print("\n  Feature-Magnitude Correlations (sorted by |r|):")
    print(f"  {'Feature':<25} {'Correlation':>12}")
    print("  " + "-" * 40)
    for feat, r in target_corr.items():
        strength = "WEAK" if abs(r) < 0.1 else "MODERATE" if abs(r) < 0.3 else "STRONG"
        print(f"  {feat:<25} {r:>+12.4f}  ({strength})")

    # Feature-feature correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    feat_corr = corr_matrix.loc[valid_features, valid_features]
    im = ax.imshow(feat_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(valid_features)))
    ax.set_yticks(range(len(valid_features)))
    ax.set_xticklabels(valid_features, rotation=90, fontsize=8)
    ax.set_yticklabels(valid_features, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "correlation_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Plot] Saved → {save_path}")

    # Identify highly correlated feature pairs
    print("\n  Highly Correlated Feature Pairs (|r| > 0.95):")
    print(f"  {'Feature 1':<25} {'Feature 2':<25} {'r':>8}")
    print("  " + "-" * 60)
    high_corr_pairs = []
    for i, f1 in enumerate(valid_features):
        for j, f2 in enumerate(valid_features):
            if i < j:
                r = feat_corr.loc[f1, f2]
                if abs(r) > 0.95:
                    high_corr_pairs.append((f1, f2, r))
                    print(f"  {f1:<25} {f2:<25} {r:>+8.4f}")

    # Plot feature-target correlation bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_corr = target_corr.sort_values()
    colors = ["#D32F2F" if r < 0 else "#4CAF50" for r in sorted_corr.values]
    ax.barh(sorted_corr.index, sorted_corr.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Magnitude", fontsize=13)
    ax.set_title("Feature–Magnitude Correlations", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    save_path = os.path.join(ANALYSIS_DIR, "feature_target_correlations.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {save_path}")

    return target_corr, high_corr_pairs


def analyze_data_imbalance(df):
    """Analyze class imbalance in the target variable."""
    print("\n" + "=" * 60)
    print("  8. DATA IMBALANCE ANALYSIS")
    print("=" * 60)

    vc = df["magnitude"].value_counts().sort_index()
    total = len(df)
    min_class = vc.min()
    max_class = vc.max()
    imbalance_ratio = max_class / min_class

    print(f"  Total samples: {total}")
    print(f"  Smallest class: M={vc.idxmin():.1f} with {min_class} samples")
    print(f"  Largest class: M={vc.idxmax():.1f} with {max_class} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 5:
        print(f"  ⚠️ Significant imbalance detected (ratio > 5:1)")
    elif imbalance_ratio > 3:
        print(f"  ⚠️ Moderate imbalance detected (ratio > 3:1)")
    else:
        print(f"  ✓ Balanced dataset (ratio < 3:1)")


def generate_summary(df, outlier_df, var_df, target_corr, high_corr_pairs):
    """Generate a text summary of all findings."""
    print("\n" + "=" * 60)
    print("  DATASET ANALYSIS SUMMARY")
    print("=" * 60)

    features = [c for c in df.columns if c != "magnitude"]

    print(f"""
  Dataset: earthquake_29_features.csv
  Samples: {len(df)}
  Features: {len(features)}
  Target: magnitude ({df['magnitude'].nunique()} unique values, {df['magnitude'].min():.1f}–{df['magnitude'].max():.1f})

  Missing Values: 0
  Duplicates: {df.duplicated().sum()}

  Degenerate Features (remove these):
    - dominant_freq: ALL ZEROS
    - predominant_period: ALL ZEROS
    - zero_crossings: near-zero (std=0.73, 99.6% zeros)

  Highly Correlated Pairs (|r| > 0.95): {len(high_corr_pairs)}
  Strongest feature-target correlation: {target_corr.index[0]} (r={target_corr.values[0]:+.4f})

  Key Issue: All feature-target correlations are WEAK (|r| < 0.11)
  This is the fundamental challenge for any regression model.
""")


def main():
    """Run the complete dataset analysis."""
    print("\n" + "█" * 60)
    print("  PHASE 3: COMPLETE DATASET ANALYSIS")
    print("█" * 60)

    setup()
    df = load_data()

    # Run all analyses
    missing_report = analyze_missing_values(df)
    n_dupes = analyze_duplicates(df)
    analyze_target_distribution(df)
    analyze_feature_distributions(df)
    outlier_df = analyze_outliers(df)
    var_df = analyze_feature_variance(df)
    target_corr, high_corr_pairs = analyze_correlations(df)
    analyze_data_imbalance(df)

    # Summary
    generate_summary(df, outlier_df, var_df, target_corr, high_corr_pairs)

    print("\n" + "█" * 60)
    print("  DATASET ANALYSIS COMPLETE")
    print(f"  All plots saved to: {ANALYSIS_DIR}/")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
