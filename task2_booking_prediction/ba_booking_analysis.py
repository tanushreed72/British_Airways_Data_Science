"""
British Airways Customer Booking Prediction
============================================
Full pipeline: EDA → Feature Engineering → RandomForest → Evaluation → Visualisations
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import os

# ── Style ─────────────────────────────────────────────────────────────────────
BA_NAVY   = "#001F5B"
BA_RED    = "#C01820"
BA_BLUE   = "#3D85C8"
BA_LTBLUE = "#D6E4F0"
PALETTE   = [BA_NAVY, BA_RED, BA_BLUE, "#E8A838", "#2ECC71", "#9B59B6"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

FIGURES = "figures"
os.makedirs(FIGURES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv("customer_booking.csv", encoding="latin1")
print(f"  Shape        : {df.shape}")
print(f"  Columns      : {df.columns.tolist()}")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"  Target dist  :\n{df['booking_complete'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- FIG 1: Target balance -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("British Airways — Booking Dataset Overview", fontsize=15, fontweight="bold", color=BA_NAVY)

counts = df["booking_complete"].value_counts()
bars = axes[0].bar(["Not Booked (0)", "Booked (1)"], counts.values,
                   color=[BA_BLUE, BA_RED], edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
axes[0].set_title("Target Class Distribution")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, 48000)

# Sales channel vs booking
chan_book = df.groupby("sales_channel")["booking_complete"].mean() * 100
axes[1].bar(chan_book.index, chan_book.values, color=[BA_NAVY, BA_RED])
axes[1].set_title("Booking Rate by Sales Channel")
axes[1].set_ylabel("Booking Rate (%)")
for i, (idx, v) in enumerate(chan_book.items()):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

# Trip type vs booking
trip_book = df.groupby("trip_type")["booking_complete"].mean() * 100
axes[2].bar(trip_book.index, trip_book.values, color=PALETTE[:3])
axes[2].set_title("Booking Rate by Trip Type")
axes[2].set_ylabel("Booking Rate (%)")
for i, (idx, v) in enumerate(trip_book.items()):
    axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{FIGURES}/01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_dataset_overview.png")

# --- FIG 2: Numeric distributions -----------------------------------------
num_cols = ["purchase_lead", "length_of_stay", "flight_hour", "flight_duration",
            "num_passengers"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Distribution of Numeric Features by Booking Outcome", fontsize=13, fontweight="bold", color=BA_NAVY)
axes = axes.flatten()

for i, col in enumerate(num_cols):
    for val, color, label in [(0, BA_BLUE, "Not Booked"), (1, BA_RED, "Booked")]:
        axes[i].hist(df[df["booking_complete"] == val][col], bins=40, alpha=0.6,
                     color=color, label=label, density=True, edgecolor="none")
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel("Density")

axes[5].axis("off")
plt.tight_layout()
fig.savefig(f"{FIGURES}/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_numeric_distributions.png")

# --- FIG 3: Booking rate by day & hour -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Temporal Patterns in Booking Behaviour", fontsize=13, fontweight="bold", color=BA_NAVY)

day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_rate = df.groupby("flight_day")["booking_complete"].mean().reindex(day_order) * 100
axes[0].bar(day_rate.index, day_rate.values, color=BA_NAVY)
axes[0].set_title("Booking Rate by Day of Week")
axes[0].set_ylabel("Booking Rate (%)")

hour_rate = df.groupby("flight_hour")["booking_complete"].mean() * 100
axes[1].plot(hour_rate.index, hour_rate.values, color=BA_RED, linewidth=2.5, marker="o", markersize=4)
axes[1].fill_between(hour_rate.index, hour_rate.values, alpha=0.15, color=BA_RED)
axes[1].set_title("Booking Rate by Flight Hour")
axes[1].set_ylabel("Booking Rate (%)")
axes[1].set_xlabel("Hour of Day")

plt.tight_layout()
fig.savefig(f"{FIGURES}/03_temporal_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_temporal_patterns.png")

# --- FIG 4: Add-ons correlation ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
addon_cols = ["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"]
addon_book = pd.DataFrame({
    col: df.groupby(col)["booking_complete"].mean() * 100
    for col in addon_cols
}).T
addon_book.columns = ["Didn't Request", "Requested"]
addon_book.index = ["Extra Baggage", "Preferred Seat", "In-Flight Meals"]

x = np.arange(len(addon_book))
w = 0.35
bars1 = ax.bar(x - w/2, addon_book["Didn't Request"], w, label="Didn't Request", color=BA_BLUE)
bars2 = ax.bar(x + w/2, addon_book["Requested"],     w, label="Requested",      color=BA_RED)
ax.set_xticks(x); ax.set_xticklabels(addon_book.index)
ax.set_ylabel("Booking Rate (%)")
ax.set_title("Add-on Requests vs Booking Completion", fontweight="bold", color=BA_NAVY)
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(f"{FIGURES}/04_addon_booking_rates.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_addon_booking_rates.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

df_model = df.copy()

# 3a. Encode flight_day as ordered integer (Mon=1 … Sun=7)
day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
df_model["flight_day_num"] = df_model["flight_day"].map(day_map)
df_model["is_weekend"] = df_model["flight_day"].isin(["Sat", "Sun"]).astype(int)

# 3b. Binary encode sales_channel & trip_type
df_model["is_mobile"] = (df_model["sales_channel"] == "Mobile").astype(int)
df_model["is_roundtrip"] = (df_model["trip_type"] == "RoundTrip").astype(int)

# 3c. Interaction: total add-ons requested
df_model["total_addons"] = (df_model["wants_extra_baggage"] +
                            df_model["wants_preferred_seat"] +
                            df_model["wants_in_flight_meals"])

# 3d. Log-transform right-skewed features
df_model["log_purchase_lead"] = np.log1p(df_model["purchase_lead"])
df_model["log_length_of_stay"] = np.log1p(df_model["length_of_stay"])

# 3e. Route frequency encode (high-freq routes may differ in booking pattern)
route_freq = df_model["route"].value_counts() / len(df_model)
df_model["route_freq"] = df_model["route"].map(route_freq)

# 3f. Booking origin frequency encode
origin_freq = df_model["booking_origin"].value_counts() / len(df_model)
df_model["origin_freq"] = df_model["booking_origin"].map(origin_freq)

# 3g. Flight peak-hour flag (06-09 early morning, 17-20 evening rush)
df_model["is_peak_hour"] = df_model["flight_hour"].apply(
    lambda h: 1 if (6 <= h <= 9) or (17 <= h <= 20) else 0
)

# 3h. Passengers × lead time interaction
df_model["pax_lead_interaction"] = df_model["num_passengers"] * df_model["log_purchase_lead"]

# Final feature list
FEATURES = [
    "num_passengers", "purchase_lead", "length_of_stay", "flight_hour",
    "flight_duration", "wants_extra_baggage", "wants_preferred_seat",
    "wants_in_flight_meals",
    # engineered
    "flight_day_num", "is_weekend", "is_mobile", "is_roundtrip",
    "total_addons", "log_purchase_lead", "log_length_of_stay",
    "route_freq", "origin_freq", "is_peak_hour", "pax_lead_interaction"
]

TARGET = "booking_complete"
X = df_model[FEATURES]
y = df_model[TARGET]

print(f"  Feature count : {len(FEATURES)}")
print(f"  Features      : {FEATURES}")
print(f"  Class balance : {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING — Random Forest with Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TRAINING RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

print("  Running 5-fold stratified cross-validation …")
cv_results = cross_validate(rf, X, y, cv=cv, scoring=scoring, return_train_score=True)

print("\n  ── Cross-Validation Results (5-fold) ──")
for metric in scoring:
    vals = cv_results[f"test_{metric}"]
    print(f"  {metric:12s}: {vals.mean():.4f} ± {vals.std():.4f}  "
          f"[{vals.min():.4f} – {vals.max():.4f}]")

# Out-of-fold predictions for confusion matrix / ROC
y_pred_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n  ── Classification Report (OOF Predictions) ──")
print(classification_report(y, y_pred, target_names=["Not Booked", "Booked"]))
print(f"  ROC-AUC : {roc_auc_score(y, y_pred_proba):.4f}")
print(f"  Avg Prec: {average_precision_score(y, y_pred_proba):.4f}")

# Fit on full dataset for feature importance
rf.fit(X, y)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. GENERATING EVALUATION VISUALISATIONS")
print("=" * 60)

# --- FIG 5: CV metric distribution ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("5-Fold Cross-Validation Performance", fontsize=13, fontweight="bold", color=BA_NAVY)

metrics_to_plot = ["accuracy", "roc_auc", "f1", "precision", "recall"]
means = [cv_results[f"test_{m}"].mean() for m in metrics_to_plot]
stds  = [cv_results[f"test_{m}"].std()  for m in metrics_to_plot]
labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]

bars = axes[0].barh(labels, means, xerr=stds, color=BA_NAVY, alpha=0.85,
                    error_kw={"ecolor": BA_RED, "capsize": 4, "linewidth": 1.5})
axes[0].set_xlim(0, 1.05)
axes[0].set_title("Mean ± Std Dev (5 folds)")
axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars, means):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
            xticklabels=["Not Booked", "Booked"],
            yticklabels=["Not Booked", "Booked"],
            linewidths=0.5, linecolor="white", cbar_kws={"label": "%"})
for i in range(2):
    for j in range(2):
        axes[1].text(j+0.5, i+0.7, f"n={cm[i,j]:,}", ha="center", fontsize=8, color="gray")
axes[1].set_title("Confusion Matrix (OOF) — Row %")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
fig.savefig(f"{FIGURES}/05_cv_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_cv_performance.png")

# --- FIG 6: ROC + Precision-Recall curves ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Discrimination Curves (OOF)", fontsize=13, fontweight="bold", color=BA_NAVY)

fpr, tpr, _ = roc_curve(y, y_pred_proba)
auc_val = roc_auc_score(y, y_pred_proba)
axes[0].plot(fpr, tpr, color=BA_NAVY, linewidth=2.5, label=f"RF  AUC = {auc_val:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
axes[0].fill_between(fpr, tpr, alpha=0.08, color=BA_NAVY)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(fontsize=10)

prec, rec, _ = precision_recall_curve(y, y_pred_proba)
ap = average_precision_score(y, y_pred_proba)
axes[1].plot(rec, prec, color=BA_RED, linewidth=2.5, label=f"RF  AP = {ap:.3f}")
baseline = y.mean()
axes[1].axhline(baseline, color="gray", linestyle="--", linewidth=1,
                label=f"Baseline  ({baseline:.2f})")
axes[1].fill_between(rec, prec, alpha=0.08, color=BA_RED)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(fontsize=10)

plt.tight_layout()
fig.savefig(f"{FIGURES}/06_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_roc_pr_curves.png")

# --- FIG 7: Feature importance (MAIN deliverable) -------------------------
importances = rf.feature_importances_
feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True)

feature_labels = {
    "num_passengers":        "Number of Passengers",
    "purchase_lead":         "Purchase Lead Time (raw)",
    "length_of_stay":        "Length of Stay (raw)",
    "flight_hour":           "Flight Hour",
    "flight_duration":       "Flight Duration",
    "wants_extra_baggage":   "Wants Extra Baggage",
    "wants_preferred_seat":  "Wants Preferred Seat",
    "wants_in_flight_meals": "Wants In-Flight Meals",
    "flight_day_num":        "Flight Day (numeric)",
    "is_weekend":            "Is Weekend Flight",
    "is_mobile":             "Mobile Channel",
    "is_roundtrip":          "Is Round Trip",
    "total_addons":          "Total Add-ons Requested",
    "log_purchase_lead":     "Log Purchase Lead Time ★",
    "log_length_of_stay":    "Log Length of Stay ★",
    "route_freq":            "Route Frequency ★",
    "origin_freq":           "Booking Origin Freq ★",
    "is_peak_hour":          "Is Peak Hour ★",
    "pax_lead_interaction":  "Pax × Lead Interaction ★",
}
feat_df["label"] = feat_df["feature"].map(feature_labels)

# Color: engineered features in red
feat_df["color"] = feat_df["feature"].apply(
    lambda f: BA_RED if "★" in feature_labels.get(f, "") else BA_NAVY
)

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(feat_df["label"], feat_df["importance"],
               color=feat_df["color"].values, alpha=0.85, edgecolor="none")
ax.set_title("Random Forest Feature Importance\n(★ = engineered feature)",
             fontsize=14, fontweight="bold", color=BA_NAVY, pad=12)
ax.set_xlabel("Importance Score (Mean Decrease Impurity)")

for bar, val in zip(bars, feat_df["importance"].values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5)

legend_handles = [
    mpatches.Patch(color=BA_NAVY, label="Original feature"),
    mpatches.Patch(color=BA_RED,  label="Engineered feature"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
ax.set_xlim(0, feat_df["importance"].max() * 1.15)
plt.tight_layout()
fig.savefig(f"{FIGURES}/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_feature_importance.png")

# --- FIG 8: Top feature deep-dives ----------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Top Feature Deep-Dives vs Booking Outcome", fontsize=13,
             fontweight="bold", color=BA_NAVY)

# purchase_lead vs booking
for val, color, lbl in [(0, BA_BLUE, "Not Booked"), (1, BA_RED, "Booked")]:
    axes[0, 0].hist(df[df["booking_complete"] == val]["purchase_lead"],
                    bins=50, alpha=0.6, density=True, color=color, label=lbl)
axes[0, 0].set_title("Purchase Lead Time")
axes[0, 0].set_xlabel("Days before flight")
axes[0, 0].legend()

# flight_duration vs booking
for val, color, lbl in [(0, BA_BLUE, "Not Booked"), (1, BA_RED, "Booked")]:
    axes[0, 1].hist(df[df["booking_complete"] == val]["flight_duration"],
                    bins=30, alpha=0.6, density=True, color=color, label=lbl)
axes[0, 1].set_title("Flight Duration")
axes[0, 1].set_xlabel("Hours")
axes[0, 1].legend()

# route_freq distribution by booking
for val, color, lbl in [(0, BA_BLUE, "Not Booked"), (1, BA_RED, "Booked")]:
    axes[1, 0].hist(df_model[df_model["booking_complete"] == val]["route_freq"],
                    bins=40, alpha=0.6, density=True, color=color, label=lbl)
axes[1, 0].set_title("Route Frequency (engineered)")
axes[1, 0].set_xlabel("Proportion of flights on this route")
axes[1, 0].legend()

# total_addons vs booking rate
addon_rate = df_model.groupby("total_addons")["booking_complete"].agg(["mean", "count"])
axes[1, 1].bar(addon_rate.index, addon_rate["mean"] * 100,
               color=[BA_NAVY, BA_BLUE, BA_RED, "#E8A838"], edgecolor="white")
axes[1, 1].set_title("Booking Rate by Total Add-ons")
axes[1, 1].set_xlabel("Number of Add-ons Requested")
axes[1, 1].set_ylabel("Booking Rate (%)")
for i, (idx, row) in enumerate(addon_rate.iterrows()):
    axes[1, 1].text(idx, row["mean"] * 100 + 0.5,
                    f"{row['mean']*100:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(f"{FIGURES}/08_feature_deepdives.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_feature_deepdives.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE SUMMARY STATS TO CSV (for PPT reference)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SAVING SUMMARY RESULTS")
print("=" * 60)

summary = {
    "metric": ["Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall",
                "Avg Precision"],
    "mean":   [cv_results["test_accuracy"].mean(),
                cv_results["test_roc_auc"].mean(),
                cv_results["test_f1"].mean(),
                cv_results["test_precision"].mean(),
                cv_results["test_recall"].mean(),
                average_precision_score(y, y_pred_proba)],
    "std":    [cv_results["test_accuracy"].std(),
                cv_results["test_roc_auc"].std(),
                cv_results["test_f1"].std(),
                cv_results["test_precision"].std(),
                cv_results["test_recall"].std(),
                0.0],
}
pd.DataFrame(summary).to_csv("outputs/model_metrics.csv", index=False)
feat_df[["feature", "label", "importance"]].sort_values("importance", ascending=False).to_csv(
    "outputs/feature_importances.csv", index=False)

print("  Saved: outputs/model_metrics.csv")
print("  Saved: outputs/feature_importances.csv")

print("\n" + "=" * 60)
print("✅  ANALYSIS COMPLETE — all figures saved to ./figures/")
print("=" * 60)
