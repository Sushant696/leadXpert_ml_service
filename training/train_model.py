"""
LeadXpert ML Training Pipeline
================================
Trains logistic regression + random forest classifiers on the synthetic
LeadXpert dataset, applies SMOTE for class imbalance, evaluates against
random and rule-based baselines, and exports the best model as a .pkl file.

Matches exactly what Objective 3 & 4 of the thesis proposal commits to:
  - Logistic regression + random forest
  - SMOTE balancing
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Benchmarked against random baseline + rule-based scoring

Author: LeadXpert Thesis — Sushant Babu Prasai
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─── Config
DATA_PATH   = "leadxpert_leads.csv"
OUTPUT_DIR  = "ml_output"
# Serving artifacts go straight into the Flask microservice's models/ dir,
# using the exact filenames app.py loads (best_model.pkl, scaler.pkl,
# encoders.pkl, feature_meta.json).
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "ml-service", "models")
SEED        = 42
TEST_SIZE   = 0.20
CAL_SIZE    = 0.15   # fraction of the training data held out (natural prevalence) for probability calibration
CV_FOLDS    = 5
CALIBRATION_METHODS = ["sigmoid", "isotonic"]  # both are tried; the one landing closest to the true base rate wins

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("\n" + "="*60)
print("  LeadXpert ML Training Pipeline")
print("="*60)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape     : {df.shape}")
print(f"  Converted : {df['converted'].sum():,} ({df['converted'].mean()*100:.2f}%)")
print(f"  Columns   : {list(df.columns)}")

# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[2/7] Preprocessing...")

# Drop identifier — not a feature
df = df.drop(columns=["lead_id"])

# Encode categoricals
cat_cols = ["lead_source", "business_vertical", "human_priority"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# Features and label
FEATURE_COLS = [c for c in df.columns if c != "converted"]
X = df[FEATURE_COLS].values
y = df["converted"].values

print(f"\n  Features  : {len(FEATURE_COLS)}")
print(f"  Feature list: {FEATURE_COLS}")

# ══════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n[3/7] Splitting 80/20...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# Carve a natural-prevalence calibration set out of the training data BEFORE
# SMOTE. CalibratedClassifierCV must be fit on data with the true ~22% base
# rate here — if it were fit on the SMOTE-balanced 50/50 data, the calibrated
# probabilities would center on 0.5 and we'd be right back to the overconfident
# 60-70 scores. The test set stays untouched for final evaluation.
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=CAL_SIZE, random_state=SEED,
    stratify=y_train_full
)
print(f"  Train : {X_train.shape[0]:,}  (converted: {y_train.sum():,}, {y_train.mean()*100:.1f}%)")
print(f"  Calib : {X_cal.shape[0]:,}  (converted: {y_cal.sum():,}, {y_cal.mean()*100:.1f}%)  [natural prevalence, for calibration]")
print(f"  Test  : {X_test.shape[0]:,}  (converted: {y_test.sum():,}, {y_test.mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# 4. SMOTE (on train only — never touch test set)
# ══════════════════════════════════════════════════════════════
print("\n[4/7] Applying SMOTE to training set...")
smote = SMOTE(random_state=SEED, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE: {len(X_train):,} (pos: {y_train.sum():,})")
print(f"  After  SMOTE: {len(X_train_bal):,} (pos: {y_train_bal.sum():,})")
print(f"  Balance ratio: {y_train_bal.mean()*100:.1f}% positive")

# Scale features (fit on SMOTE'd train, transform all splits)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_cal_scaled   = scaler.transform(X_cal)
X_test_scaled  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n[5/7] Training models...")

# ── Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr.fit(X_train_scaled, y_train_bal)

# ── Random Forest
# Initial training pass: keep class_weight balanced alongside SMOTE so the
# model can be benchmarked before the calibration fix lands.
print("  Training Random Forest (SMOTE-balanced, class_weight='balanced')...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train_bal)

# ── Random Baseline (DummyClassifier — stratified random)
print("  Fitting Random Baseline...")
dummy = DummyClassifier(strategy="stratified", random_state=SEED)
dummy.fit(X_train_scaled, y_train_bal)

# ══════════════════════════════════════════════════════════════
# 5b. FIX 2 — CALIBRATE THE RANDOM FOREST PROBABILITIES
# ══════════════════════════════════════════════════════════════
# The SMOTE-balanced RF is a good ranker (AUC) but its raw predict_proba is not
# calibrated to the real ~22% base rate. Wrap the prefit RF in
# CalibratedClassifierCV and fit the calibration map on the held-out,
# natural-prevalence calibration set (cv="prefit" → the base RF is NOT refit).
# We try both sigmoid (Platt) and isotonic and keep whichever lands the mean
# predicted probability closest to the calibration set's true base rate.
print("\n  Calibrating Random Forest probabilities...")
cal_base_rate = float(y_cal.mean())
calibrators = {}
for method in CALIBRATION_METHODS:
    cc = CalibratedClassifierCV(rf, method=method, cv="prefit")
    cc.fit(X_cal_scaled, y_cal)
    mean_pred = float(cc.predict_proba(X_cal_scaled)[:, 1].mean())
    calibrators[method] = {"model": cc, "mean_pred": mean_pred}
    print(f"    {method:9s}: mean predicted prob on cal set = {mean_pred*100:.2f}% "
          f"(target base rate {cal_base_rate*100:.2f}%)")

CALIBRATION_METHOD = min(
    CALIBRATION_METHODS,
    key=lambda m: abs(calibrators[m]["mean_pred"] - cal_base_rate),
)
calibrated_rf = calibrators[CALIBRATION_METHOD]["model"]
print(f"  → Chosen calibration method: {CALIBRATION_METHOD} "
      f"(closest to base rate)")

# ══════════════════════════════════════════════════════════════
# 6. RULE-BASED BASELINE
# ══════════════════════════════════════════════════════════════
# Simulates how a sales agent currently prioritizes:
# "High" if referral source OR stage_probability >= 60 OR activity_count >= 6
# Matches human cognitive heuristic described in the thesis lit review

def rule_based_score(X_arr, feature_names):
    """
    Simple rule-based classifier mimicking human SME sales agent behaviour.
    Returns binary predictions (1 = predicted to convert).
    """
    df_x = pd.DataFrame(X_arr, columns=feature_names)
    # Decode stage_probability from scaled — use raw test set instead
    # We'll pass raw X_test for this baseline
    pred = np.zeros(len(df_x), dtype=int)
    # Rules (on scaled values, so use relative thresholds via percentile)
    for i, row in df_x.iterrows():
        # Rule 1: stage_probability feature is high (top 40%)
        stage_prob_idx = feature_names.index("stage_probability")
        activity_idx   = feature_names.index("activity_count")
        rotten_idx     = feature_names.index("is_rotten")
        # In scaled space: positive z-score means above mean
        if (row.iloc[stage_prob_idx] > 0.5 or
            row.iloc[activity_idx] > 0.5) and \
           row.iloc[rotten_idx] < 0:
            pred[i] = 1
    return pred

# Apply rule-based on raw (unscaled) test — simpler and more interpretable
def rule_based_raw(X_raw, feature_names):
    df_x = pd.DataFrame(X_raw, columns=feature_names)
    stage_prob = df_x["stage_probability"]
    activity   = df_x["activity_count"]
    is_rotten  = df_x["is_rotten"]
    # Rule: predict convert if stage_probability >= 60 AND activity >= 5 AND not rotten
    pred = (
        (stage_prob >= 60) &
        (activity >= 5) &
        (is_rotten == 0)
    ).astype(int)
    return pred.values

y_pred_rule = rule_based_raw(X_test, FEATURE_COLS)

# ══════════════════════════════════════════════════════════════
# 7. EVALUATE
# ══════════════════════════════════════════════════════════════
print("\n[6/7] Evaluating all models...")

def evaluate(name, y_true, y_pred, y_proba=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc_roc": auc}

results = []

# Logistic Regression
lr_pred  = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("Logistic Regression", y_test, lr_pred, lr_proba))

# Random Forest — the DEPLOYED model is the calibrated wrapper. AUC-ROC is
# preserved under monotonic calibration; accuracy/F1/recall at the 0.5 threshold
# shift because probabilities are now centered near the true base rate instead
# of inflated. We also keep the raw RF AUC to show discrimination is unchanged.
rf_proba_raw = rf.predict_proba(X_test_scaled)[:, 1]
rf_auc_raw   = roc_auc_score(y_test, rf_proba_raw)
rf_pred  = calibrated_rf.predict(X_test_scaled)
rf_proba = calibrated_rf.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("Random Forest", y_test, rf_pred, rf_proba))

# Random Baseline
dummy_pred  = dummy.predict(X_test_scaled)
dummy_proba = dummy.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate("Random Baseline", y_test, dummy_pred, dummy_proba))

# Rule-Based Baseline (no proba)
results.append(evaluate("Rule-Based Baseline", y_test, y_pred_rule))

results_df = pd.DataFrame(results).set_index("model")
print("\n  ┌─────────────────────────────────────────────────────────────────┐")
print("  │                   MODEL EVALUATION RESULTS                      │")
print("  ├──────────────────────┬──────────┬──────────┬──────────┬─────────┤")
print("  │ Model                │ Accuracy │ F1 Score │ AUC-ROC  │ Recall  │")
print("  ├──────────────────────┼──────────┼──────────┼──────────┼─────────┤")
for model, row in results_df.iterrows():
    auc_str = f"{row['auc_roc']:.4f}" if row['auc_roc'] else "  N/A   "
    print(f"  │ {model:<20} │  {row['accuracy']:.4f}  │  {row['f1']:.4f}  │ {auc_str} │ {row['recall']:.4f}  │")
print("  └──────────────────────┴──────────┴──────────┴──────────┴─────────┘")

# 5-Fold Cross Validation on best model (RF)
print("\n  5-Fold Cross Validation (Random Forest)...")
cv_kfold = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_scores = cross_validate(
    rf, X_train_scaled, y_train_bal,
    cv=cv_kfold,
    scoring=["accuracy", "f1", "roc_auc"],
    n_jobs=-1
)
print(f"  CV Accuracy : {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
print(f"  CV F1       : {cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}")
print(f"  CV AUC-ROC  : {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")

# LR cross-validated AUC (for the regenerated model_comparison.json)
print("\n  5-Fold Cross Validation (Logistic Regression)...")
cv_lr_auc = cross_val_score(
    lr, X_train_scaled, y_train_bal,
    cv=cv_kfold, scoring="roc_auc", n_jobs=-1
)
print(f"  CV AUC-ROC  : {cv_lr_auc.mean():.4f} ± {cv_lr_auc.std():.4f}")

# Classification report
print("\n  Random Forest — Full Classification Report:")
print(classification_report(y_test, rf_pred, target_names=["Not Converted", "Converted"]))

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
print("\n[7/7] Generating plots...")
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LeadXpert ML Model Evaluation", fontsize=16, fontweight="bold")

# ── Plot 1: Metric Comparison Bar Chart
ax1 = axes[0, 0]
metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(metrics))
w = 0.20
models_plot = ["Logistic Regression", "Random Forest", "Random Baseline", "Rule-Based Baseline"]
colors = ["#4F86C6", "#2ECC71", "#E74C3C", "#F39C12"]
for i, (model, color) in enumerate(zip(models_plot, colors)):
    vals = [results_df.loc[model, m] for m in metrics]
    ax1.bar(x + i * w, vals, w, label=model, color=color, alpha=0.85)
ax1.set_xticks(x + w * 1.5)
ax1.set_xticklabels(metric_labels)
ax1.set_ylim(0, 1.05)
ax1.set_title("Model Performance Comparison")
ax1.legend(fontsize=7)
ax1.set_ylabel("Score")

# ── Plot 2: ROC Curves
ax2 = axes[0, 1]
for name, proba, color in [
    ("Logistic Regression", lr_proba, "#4F86C6"),
    ("Random Forest", rf_proba, "#2ECC71"),
    ("Random Baseline", dummy_proba, "#E74C3C"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax2.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves")
ax2.legend(fontsize=8)

# ── Plot 3: Random Forest Confusion Matrix
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax3,
            xticklabels=["Not Converted", "Converted"],
            yticklabels=["Not Converted", "Converted"])
ax3.set_title("Random Forest — Confusion Matrix")
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

# ── Plot 4: Feature Importance (RF)
ax4 = axes[1, 1]
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
colors_fi = ["#2ECC71" if v > feat_imp.median() else "#95A5A6" for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax4, color=colors_fi)
ax4.set_title("Feature Importances (Random Forest)")
ax4.set_xlabel("Importance Score")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "evaluation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {plot_path}")

# ══════════════════════════════════════════════════════════════
# FRESH-LEAD VALIDATION (critical check — see prompt.md step 4)
# ══════════════════════════════════════════════════════════════
# Build hand-crafted "day-one" leads (all-zero history, NEW stage,
# stage_probability=20) across a spread of lead sources and run them through the
# calibrated model exactly the way app.py will (encode → order → scale →
# predict_proba). Scores should now land near the true ~22% base rate, spread by
# source (lead_source is the #1 feature), and nothing should default to 60-70.
print("\n  Fresh-lead validation (calibrated model)...")

def _make_fresh(source: str) -> dict:
    return {
        "lead_source":             source,
        "business_vertical":       "GENERAL",
        "human_priority":          "MEDIUM",
        "lead_value":              0.0,
        "days_in_pipeline":        0,
        "time_in_current_stage":   0,
        "days_since_last_contact": 0,
        "activity_count":          0,
        "task_count":              0,
        "note_count":              0,
        "stage_index":             0,
        "stage_probability":       20,
        "is_rotten":               0,
        "has_upcoming_task":       0,
    }

fresh_sources = ["REFERRAL", "WALK_IN", "WHATSAPP", "WEBSITE",
                 "FACEBOOK", "COLD_OUTREACH"]
fresh_rows = pd.DataFrame([_make_fresh(s) for s in fresh_sources])
for col in cat_cols:  # encode categoricals with the fitted LabelEncoders
    fresh_rows[col] = encoders[col].transform(fresh_rows[col].astype(str))
fresh_scaled = scaler.transform(fresh_rows[FEATURE_COLS].values)
fresh_scores = calibrated_rf.predict_proba(fresh_scaled)[:, 1] * 100
fresh_validation = [
    {"lead_source": s, "score": round(float(sc), 1)}
    for s, sc in zip(fresh_sources, fresh_scores)
]
print("  ┌──────────────────┬─────────┐")
print("  │ Fresh lead source│  Score  │")
print("  ├──────────────────┼─────────┤")
for row in fresh_validation:
    print(f"  │ {row['lead_source']:<16} │  {row['score']:>5.1f}  │")
print("  └──────────────────┴─────────┘")
print(f"  Mean fresh-lead score: {fresh_scores.mean():.1f} "
      f"(true base rate ≈ {y_test.mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# EXPORT ARTIFACTS
# ══════════════════════════════════════════════════════════════
# Best servable model chosen by AUC-ROC (discrimination), NOT F1: calibration
# deliberately shifts probabilities toward the true base rate, which lowers the
# 0.5-threshold F1/accuracy while leaving ranking power (AUC) intact — F1 would
# wrongly punish the calibrated RF. The exported RF is the CALIBRATED wrapper
# (what Flask serves); LR is exported raw.
SERVABLE_MODELS = {"Logistic Regression": lr, "Random Forest": rf}
best_model_name = results_df.loc[list(SERVABLE_MODELS.keys()), "auc_roc"].idxmax()
best_model = calibrated_rf if best_model_name == "Random Forest" else lr
print(f"\n  Best model by AUC-ROC: {best_model_name} "
      f"(exported {'calibrated' if best_model_name == 'Random Forest' else 'raw'})")

# Raw RF feature importances (aligned to FEATURE_COLS). Persisted here because
# the CalibratedClassifierCV wrapper exposes neither feature_importances_ nor
# coef_, so app.py reads these from feature_meta.json for the /score topFeatures.
rf_importances = [round(float(v), 6) for v in rf.feature_importances_]

# ── Thesis artifacts (full set) → ml_output/
joblib.dump(rf,           os.path.join(OUTPUT_DIR, "random_forest.pkl"))
joblib.dump(calibrated_rf, os.path.join(OUTPUT_DIR, "random_forest_calibrated.pkl"))
joblib.dump(lr,           os.path.join(OUTPUT_DIR, "logistic_regression.pkl"))
joblib.dump(scaler,       os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(encoders,     os.path.join(OUTPUT_DIR, "label_encoders.pkl"))

# Feature metadata (critical — Flask needs this exact column order + the keys
# app.py reads: feature_cols, categorical_cols, best_model, feature_importances).
feature_meta = {
    "feature_cols":        FEATURE_COLS,
    "categorical_cols":    cat_cols,
    "target_col":          "converted",
    "best_model":          best_model_name,
    "seed":                SEED,
    "smote_applied":       True,
    "class_weight":        None,
    "calibration":         {"method": CALIBRATION_METHOD, "cv": "prefit", "cal_size": CAL_SIZE},
    "feature_importances": rf_importances,
    "cv_folds":            CV_FOLDS,
    "test_size":           TEST_SIZE,
}
with open(os.path.join(OUTPUT_DIR, "feature_meta.json"), "w") as f:
    json.dump(feature_meta, f, indent=2)

# ── Serving artifacts → ml-service/models/ (exact filenames app.py loads).
# best_model.pkl is the calibrated RF; it exposes predict_proba with the same
# column order/shape app.py already sends, so the /score contract is unchanged.
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(rf,         os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(lr,         os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(scaler,     os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(encoders,   os.path.join(MODELS_DIR, "encoders.pkl"))
with open(os.path.join(MODELS_DIR, "feature_meta.json"), "w") as f:
    json.dump(feature_meta, f, indent=2)
print(f"  Serving artifacts → {os.path.normpath(MODELS_DIR)}")

# ── Regenerate model_comparison.json (all models, apples-to-apples)
comparison = {
    "results": {
        model: {
            "accuracy":  round(float(row["accuracy"]), 4),
            "precision": round(float(row["precision"]), 4),
            "recall":    round(float(row["recall"]), 4),
            "f1":        round(float(row["f1"]), 4),
            "auc_roc":   (round(float(row["auc_roc"]), 4)
                          if pd.notna(row["auc_roc"]) else None),
        }
        for model, row in results_df.iterrows()
    },
    "cv_lr": {
        "mean":   round(float(cv_lr_auc.mean()), 4),
        "std":    round(float(cv_lr_auc.std()), 4),
        "scores": [float(s) for s in cv_lr_auc],
    },
    "cv_rf": {
        "mean":   round(float(cv_scores["test_roc_auc"].mean()), 4),
        "std":    round(float(cv_scores["test_roc_auc"].std()), 4),
        "scores": [float(s) for s in cv_scores["test_roc_auc"]],
    },
    "feature_importances": [
        {"feature": FEATURE_COLS[i], "importance": rf_importances[i]}
        for i in np.argsort(rf.feature_importances_)[::-1]
    ],
    "best_model": best_model_name,
    # Extra context for this fix (does not change the original schema above)
    "rf_auc_raw":         round(float(rf_auc_raw), 4),
    "rf_auc_calibrated":  round(float(results_df.loc["Random Forest", "auc_roc"]), 4),
    "calibration_method": CALIBRATION_METHOD,
    "fresh_lead_validation": fresh_validation,
    "fresh_lead_mean_score": round(float(fresh_scores.mean()), 1),
    "test_base_rate": round(float(y_test.mean()), 4),
}
COMPARISON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model_comparison.json")
with open(COMPARISON_PATH, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"  Regenerated: {COMPARISON_PATH}")

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"))

# Save CV results
cv_summary = {
    "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
    "cv_accuracy_std":  float(cv_scores["test_accuracy"].std()),
    "cv_f1_mean":       float(cv_scores["test_f1"].mean()),
    "cv_f1_std":        float(cv_scores["test_f1"].std()),
    "cv_auc_mean":      float(cv_scores["test_roc_auc"].mean()),
    "cv_auc_std":       float(cv_scores["test_roc_auc"].std()),
}
with open(os.path.join(OUTPUT_DIR, "cv_results.json"), "w") as f:
    json.dump(cv_summary, f, indent=2)

print(f"\n  Artifacts saved to ./{OUTPUT_DIR}/")
print(f"    random_forest.pkl")
print(f"    logistic_regression.pkl")
print(f"    scaler.pkl")
print(f"    label_encoders.pkl")
print(f"    feature_meta.json")
print(f"    evaluation_results.csv")
print(f"    cv_results.json")
print(f"    evaluation_plots.png")

print(f"\n{'='*60}")
print(f"  Training complete.")
print(f"{'='*60}\n")
