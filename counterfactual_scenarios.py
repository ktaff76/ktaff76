# ── DiCE Counterfactual Scenarios: Moving High-Risk Students to Low/Medium Risk ──

import dice_ml
from dice_ml import Dice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Prepare DiCE data object ────────────────────────────────────────────
# DiCE needs the ENCODED training data + outcome column
train_df = X_train.copy()
train_df["burnout_level"] = y_train

# Identify feature types for DiCE
# NOTE: age, cgpa are excluded from continuous_features — they are non-actionable
# NOTE: gender, course, year are excluded from categorical_features — they are non-actionable
continuous_features = [
    "daily_study_hours", "daily_sleep_hours", "screen_time_hours",
    "anxiety_score", "depression_score", "academic_pressure_score",
    "financial_stress_score", "social_support_score",
    "physical_activity_hours", "attendance_percentage",
]
categorical_features = [
    "stress_level", "sleep_quality", "internet_quality",
]

d = dice_ml.Data(
    dataframe=train_df,
    continuous_features=continuous_features,
    categorical_features=categorical_features,
    outcome_name="burnout_level",
)

m = dice_ml.Model(model=model, backend="sklearn")
exp = Dice(d, m, method="random")

# ─── 2. Select HIGH-RISK test instances ────────────────────────────────────
# burnout_label_map: {0: 'High', 1: 'Low', 2: 'Medium'}
high_label_code = [k for k, v in burnout_label_map.items() if v == "High"][0]

X_test_high = X_test[y_test == high_label_code].head(10)   # sample 10 high-risk
print(f"Using {len(X_test_high)} high-risk test instances for counterfactual analysis.\n")

# ─── 3. Generate counterfactuals targeting Low and Medium ──────────────────
target_labels = {
    k: v for k, v in burnout_label_map.items() if v in ("Low", "Medium")
}

# Features we allow DiCE to change (actionable / modifiable)
# age, gender, cgpa, course, and year are EXCLUDED — they are fixed/non-actionable
actionable_features = [
    "daily_study_hours", "daily_sleep_hours", "screen_time_hours",
    "anxiety_score", "depression_score", "academic_pressure_score",
    "financial_stress_score", "social_support_score",
    "physical_activity_hours", "attendance_percentage",
    "stress_level", "sleep_quality", "internet_quality",
]

# DiCE varies only the actionable features; demographics/identity are held fixed
features_to_vary = actionable_features

results = []

for idx, row in X_test_high.iterrows():
    query = row.to_frame().T.reset_index(drop=True)

    for target_code, target_name in target_labels.items():
        try:
            cf = exp.generate_counterfactuals(
                query,
                total_CFs=3,
                desired_class=target_code,
                features_to_vary=features_to_vary,
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and len(cf_df) > 0:
                for _, cf_row in cf_df.iterrows():
                    diff = {}
                    for feat in actionable_features:
                        orig_val = float(query[feat].values[0])
                        new_val  = float(cf_row[feat])
                        delta    = new_val - orig_val
                        if abs(delta) > 0.001:
                            diff[feat] = {
                                "original": round(orig_val, 2),
                                "new":      round(new_val, 2),
                                "change":   round(delta, 2),
                            }
                    results.append({
                        "student_idx":  idx,
                        "target_class": target_name,
                        "changes":      diff,
                        "n_changes":    len(diff),
                    })
        except Exception as e:
            print(f"  ⚠️  Could not generate CF for student {idx} → {target_name}: {e}")

print(f"\nTotal counterfactual scenarios generated: {len(results)}\n")

# ─── 4. Aggregate: which features are changed most often & by how much ─────
all_changes = []
for r in results:
    for feat, vals in r["changes"].items():
        all_changes.append({
            "feature":      feat,
            "target_class": r["target_class"],
            "change":       vals["change"],
            "abs_change":   abs(vals["change"]),
        })

changes_df = pd.DataFrame(all_changes)

if changes_df.empty:
    print("No counterfactual changes were captured. Try increasing total_CFs or broadening features_to_vary.")
else:
    # Frequency + mean direction of change per feature per target class
    summary = (
        changes_df
        .groupby(["feature", "target_class"])
        .agg(
            frequency    = ("change", "count"),
            mean_change  = ("change", "mean"),
            mean_abs_chg = ("abs_change", "mean"),
        )
        .reset_index()
        .sort_values(["target_class", "frequency"], ascending=[True, False])
    )

    print("=" * 60)
    print("INTERVENTION SUMMARY: Changes that move HIGH → target class")
    print("=" * 60)
    for target in ["Low", "Medium"]:
        sub = summary[summary["target_class"] == target].head(10)
        print(f"\n── Moving to {target.upper()} RISK ─────────────────────────")
        print(sub[["feature", "frequency", "mean_change", "mean_abs_chg"]].to_string(index=False))

    # ─── 5. Bar chart: top interventions per target class ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, target in zip(axes, ["Low", "Medium"]):
        sub = (
            summary[summary["target_class"] == target]
            .sort_values("frequency", ascending=False)
            .head(8)
        )
        colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in sub["mean_change"]]
        bars = ax.barh(
            sub["feature"][::-1],
            sub["mean_change"][::-1],
            color=colors[::-1],
            edgecolor="white",
        )
        # Annotate with frequency
        for bar, freq in zip(bars, sub["frequency"][::-1]):
            w = bar.get_width()
            ax.text(
                w + (0.05 if w >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                f"n={freq}",
                va="center",
                ha="left" if w >= 0 else "right",
                fontsize=8,
            )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"HIGH → {target.upper()} RISK\nMean feature change", fontsize=11)
        ax.set_xlabel("Mean Change (negative = decrease, positive = increase)")

    plt.suptitle("Counterfactual Intervention Scenarios\n(What needs to change to move high-risk students)", fontsize=13)
    plt.tight_layout()
    plt.savefig("counterfactual_interventions.jpg", dpi=300, bbox_inches="tight", format="jpg")
    plt.show()

    # ─── 6. Print a human-readable scenario for the top counterfactual ─────
    print("\n" + "=" * 60)
    print("EXAMPLE ACTIONABLE SCENARIO (first high-risk student)")
    print("=" * 60)
    first_cf = [r for r in results if r["target_class"] == "Low"]
    if first_cf:
        best = min(first_cf, key=lambda x: x["n_changes"])   # fewest changes
        print(f"\nStudent index: {best['student_idx']}  →  Target: {best['target_class']} risk")
        print(f"Number of changes required: {best['n_changes']}\n")
        for feat, vals in best["changes"].items():
            direction = "▼ decrease" if vals["change"] < 0 else "▲ increase"
            print(f"  {feat:<30}  {direction}  {vals['original']} → {vals['new']}  (Δ {vals['change']:+.2f})")
    else:
        print("No Low-risk counterfactuals found; try relaxing constraints.")
