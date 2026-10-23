"""
feature_audit.py
----------------
Automated feature quality analysis for the earthquake PINN pipeline.

Steps:
    1. Detect degenerate features (zero / near-zero variance, all-equal values).
    2. Detect highly correlated redundant features; keep physics-motivated
       representative per group.
    3. Analyse target variable distribution.
    4. Write FEATURE_ANALYSIS_REPORT.md.

Design principle:
    All functions operate ONLY on training data so that the audit itself
    does not introduce any leakage from validation or test sets.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Physics-motivated feature priority list.
# When selecting a representative from a correlated group, we prefer the
# feature that carries the most physically direct meaning for seismic
# magnitude estimation.  First match wins.
# ---------------------------------------------------------------------------
PHYSICS_PRIORITY: List[str] = [
    "energy",             # P-wave energy integral  ∫a²dt  — direct G-R proxy
    "arias",              # Arias intensity          — gold-standard seismic measure
    "arias_intensity",    # alternative column name
    "pga",                # Peak ground acceleration — most cited in seismology
    "rms",                # RMS amplitude            — seismic moment proxy
    "cav",                # Cumulative absolute vel  — engineering standard
    "spectral_centroid",  # Frequency proxy          — used in physics constraint
    "duration",           # Seismic duration         — related to stress drop
]


# ---------------------------------------------------------------------------
# 1. Degenerate feature detection
# ---------------------------------------------------------------------------

def detect_degenerate_features(
    X: pd.DataFrame,
    variance_threshold: float = 1e-6,
) -> Tuple[List[str], Dict[str, str]]:
    """Detect zero or near-zero variance features from training data.

    A feature is considered degenerate when:
      - All sample values are identical  (nunique == 1), OR
      - Sample variance < variance_threshold.

    Args:
        X:                   Training feature DataFrame only (no val/test).
        variance_threshold:  Minimum acceptable variance.

    Returns:
        Tuple of:
            features_to_remove  – list of column names.
            removal_reasons     – dict {col: human-readable reason}.
    """
    removal_reasons: Dict[str, str] = {}

    for col in X.columns:
        vals = X[col].dropna()

        if vals.nunique() <= 1:
            sample_val = vals.iloc[0] if len(vals) > 0 else "N/A"
            removal_reasons[col] = (
                f"All {len(vals)} values are identical (value={sample_val})"
            )
            continue

        var = float(vals.var())
        if var < variance_threshold:
            removal_reasons[col] = (
                f"Near-zero variance ({var:.3e} < threshold {variance_threshold:.1e})"
            )

    features_to_remove = list(removal_reasons.keys())

    print(f"\n[FeatureAudit] Degenerate scan — {len(X.columns)} features checked")
    if features_to_remove:
        print(f"  -> {len(features_to_remove)} degenerate feature(s) found:")
        for col, reason in removal_reasons.items():
            print(f"    - '{col}': {reason}")
    else:
        print("  -> No degenerate features found.")

    return features_to_remove, removal_reasons


# ---------------------------------------------------------------------------
# 2. Correlation-based redundancy analysis
# ---------------------------------------------------------------------------

def analyze_correlations(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[List[str], Dict[str, dict]]:
    """Identify and mark redundant features based on Pearson |r| correlation.

    Algorithm:
        1. Compute upper-triangle of |correlation| matrix.
        2. For each feature not yet assigned to a group, find all features
           with |r| > threshold.
        3. Within each group keep the highest-priority representative
           (per PHYSICS_PRIORITY, then alphabetical fallback).
        4. Mark the rest for removal.

    Args:
        X:          Training features (degenerate columns already removed).
        threshold:  |r| above which features are considered redundant.

    Returns:
        Tuple of:
            all_removed   – flat list of column names to drop.
            group_info    – dict {representative: {group, removed, corr, …}}.
    """
    corr_matrix = X.corr(method="pearson").abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    visited: set = set()
    group_info: Dict[str, dict] = {}
    all_removed: List[str] = []

    for col in corr_matrix.columns:
        if col in visited:
            continue

        # Features with |r| > threshold in the upper triangle
        hi_corr = upper[col][upper[col] > threshold].index.tolist()
        if not hi_corr:
            continue

        group = [col] + [c for c in hi_corr if c not in visited]
        group = list(dict.fromkeys(group))  # deduplicate, preserve order

        if len(group) <= 1:
            continue

        # Select physics-motivated representative
        representative: str = ""
        for pref in PHYSICS_PRIORITY:
            if pref in group:
                representative = pref
                break
        if not representative:
            representative = sorted(group)[0]

        to_remove = [f for f in group if f != representative]
        all_removed.extend(to_remove)
        visited.update(group)

        group_info[representative] = {
            "group":     group,
            "removed":   to_remove,
            "threshold": threshold,
            "pairwise_corr": {
                r: round(float(corr_matrix.loc[
                    representative if representative in corr_matrix.index else r,
                    r if r in corr_matrix.columns else representative
                ]), 4)
                for r in to_remove
            },
        }

    all_removed = list(dict.fromkeys(all_removed))

    print(f"\n[FeatureAudit] Correlation analysis (|r| threshold = {threshold})")
    if group_info:
        print(f"  -> {len(all_removed)} redundant feature(s) across {len(group_info)} group(s):")
        for rep, info in group_info.items():
            print(f"    Group '{rep}': removed {info['removed']}")
    else:
        print("  -> No redundant features found at this threshold.")

    return all_removed, group_info


# ---------------------------------------------------------------------------
# 3. Target distribution analysis
# ---------------------------------------------------------------------------

def analyze_target(y: pd.Series) -> Dict:
    """Analyse the distribution of the magnitude target variable.

    Checks for low unique-value count (quasi-discrete magnitudes) which
    degrades NN optimisation without target normalisation.

    Args:
        y: Target Series (earthquake magnitude values).

    Returns:
        Dictionary of descriptive statistics.
    """
    stats = {
        "count":         int(len(y)),
        "unique_values": int(y.nunique()),
        "unique_list":   sorted([round(float(v), 4) for v in y.unique()]),
        "min":           float(y.min()),
        "max":           float(y.max()),
        "mean":          float(y.mean()),
        "std":           float(y.std()),
        "median":        float(y.median()),
    }

    print("\n[FeatureAudit] Target (magnitude) distribution:")
    print(f"  Samples      : {stats['count']}")
    print(f"  Unique values: {stats['unique_values']}")
    print(f"  Range        : [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean ± Std   : {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Unique list  : {stats['unique_list']}")

    if stats["unique_values"] <= 20:
        print(
            "\n  WARNING: Only "
            + str(stats["unique_values"])
            + " unique magnitude values detected.\n"
            "     Target StandardScaler strongly recommended to normalise\n"
            "     the MSE loss landscape for NN optimisation."
        )

    return stats


# ---------------------------------------------------------------------------
# 4. Markdown report generator
# ---------------------------------------------------------------------------

def generate_feature_report(
    original_features: List[str],
    degenerate_removed: List[str],
    degenerate_reasons: Dict[str, str],
    correlation_removed: List[str],
    correlation_groups: Dict[str, dict],
    final_features: List[str],
    target_stats: Dict,
    save_path: str = "FEATURE_ANALYSIS_REPORT.md",
) -> None:
    """Write a detailed Markdown feature analysis report.

    Args:
        original_features:  All feature names before any removal.
        degenerate_removed: Features removed (degenerate statistics).
        degenerate_reasons: {col: reason} from detect_degenerate_features().
        correlation_removed:Features removed (high correlation).
        correlation_groups: {rep: info} from analyze_correlations().
        final_features:     Feature list passed to the model.
        target_stats:       Output of analyze_target().
        save_path:          Output .md file path.
    """
    L: List[str] = []

    L += [
        "# Feature Analysis Report",
        "",
        "Auto-generated by `src/feature_audit.py`.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "|---|---|",
        f"| Original feature count | {len(original_features)} |",
        f"| Removed — degenerate (zero/near-zero variance) | {len(degenerate_removed)} |",
        f"| Removed — redundant (high correlation) | {len(correlation_removed)} |",
        f"| **Final feature count used for training** | **{len(final_features)}** |",
        "",
    ]

    # Target analysis
    L += [
        "---",
        "",
        "## Target Variable (Magnitude) Analysis",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Total samples | {target_stats['count']} |",
        f"| Unique magnitude values | {target_stats['unique_values']} |",
        f"| Min | {target_stats['min']:.4f} |",
        f"| Max | {target_stats['max']:.4f} |",
        f"| Mean | {target_stats['mean']:.4f} |",
        f"| Std | {target_stats['std']:.4f} |",
        f"| Median | {target_stats['median']:.4f} |",
        f"| All unique values | `{target_stats['unique_list']}` |",
        "",
    ]
    if target_stats["unique_values"] <= 20:
        L += [
            "> **Note**: Only "
            + str(target_stats["unique_values"])
            + " unique magnitude values. "
            "Target `StandardScaler` is applied (fitted on training data only) "
            "to normalise the optimisation landscape.",
            "",
        ]

    # Degenerate features
    L += ["---", "", "## Degenerate Features Removed", ""]
    if degenerate_removed:
        L += ["| Feature | Reason |", "|---|---|"]
        for f in degenerate_removed:
            L.append(f"| `{f}` | {degenerate_reasons.get(f, 'N/A')} |")
        L.append("")
    else:
        L += ["_None detected._", ""]

    # Correlation-based removal
    L += ["---", "", "## Redundant Features Removed (High Correlation)", ""]
    if correlation_groups:
        for rep, info in correlation_groups.items():
            L += [
                f"### Group representative: `{rep}`",
                "",
                f"Threshold: |r| > {info['threshold']}  |  "
                f"Group members: `{'`, `'.join(info['group'])}`",
                "",
                f"| Removed | |r| with `{rep}` | Reason |",
                "|---|---|---|",
            ]
            for rf in info["removed"]:
                cv = info["pairwise_corr"].get(rf, "N/A")
                L.append(f"| `{rf}` | {cv} | Correlation > {info['threshold']} |")
            L.append("")
    else:
        L += ["_No features removed at the given correlation threshold._", ""]

    # Final feature set
    L += ["---", "", "## Final Feature Set", "",
          f"**{len(final_features)} features** passed to the model:", ""]
    for i, f in enumerate(final_features, 1):
        L.append(f"{i}. `{f}`")
    L.append("")

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(f"[FeatureAudit] Report saved -> {save_path}")
