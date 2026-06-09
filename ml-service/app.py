"""
LeadXpert ML Scoring Microservice
===================================
Flask REST API that wraps the trained Random Forest model.
Runs on port 5001 (internal only — not to be exposed externally).

Endpoints:
  GET  /health         — liveness check
  GET  /features       — returns expected feature schema

Run:
  python app.py                    (dev)
  gunicorn -w 2 -b 0.0.0.0:5001 app:app   (prod)
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # only allow Next.js dev origin

# ── Model artifact paths (relative to this file)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
META_PATH     = os.path.join(MODELS_DIR, "feature_meta.json")

# ── Load artifacts at startup (once)
log.info("Loading ML artifacts...")
try:
    # joblib.load reads both joblib dumps (how train_model.py saves) and
    # plain pickles, so it works regardless of how the artifacts were created.
    MODEL    = joblib.load(MODEL_PATH)
    SCALER   = joblib.load(SCALER_PATH)
    ENCODERS = joblib.load(ENCODERS_PATH)
    with open(META_PATH, "r") as f: META = json.load(f)
    FEATURE_COLS    = META["feature_cols"]
    CATEGORICAL_COLS = META["categorical_cols"]
    MODEL_NAME      = META["best_model"]
    # Feature importances are persisted in meta because the deployed model may be
    # a CalibratedClassifierCV wrapper, which exposes neither feature_importances_
    # nor coef_. Falls back to reading them off the model if meta doesn't have them.
    FEATURE_IMPORTANCES = META.get("feature_importances")
    log.info(f"Loaded model: {MODEL_NAME}")
    log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    ARTIFACTS_LOADED = True
except Exception as e:
    log.error(f"Failed to load artifacts: {e}")
    ARTIFACTS_LOADED = False

# ── Valid enum values (mirrors leadXpert_server/src/types/shared.types.ts).
# Keep in sync with training/generate_leads.py — the model is only valid for
# the values it was trained on.
VALID_LEAD_SOURCES = [
    "FACEBOOK", "INSTAGRAM", "TIKTOK", "VIBER", "WHATSAPP",
    "WALK_IN", "REFERRAL", "GOOGLE", "WEBSITE", "YOUTUBE",
    "EMAIL", "PHONE_CALL", "LINKEDIN", "EVENT", "COLD_OUTREACH", "OTHER",
]
VALID_VERTICALS = [
    "EDUCATION_CONSULTANCY", "DIGITAL_MARKETING", "IT_SOFTWARE",
    "LEGAL_FINANCIAL", "REAL_ESTATE", "RECRUITMENT",
    "EVENT_MANAGEMENT", "GENERAL",
]
VALID_PRIORITIES = ["URGENT", "HIGH", "MEDIUM", "LOW"]


# CORE SCORING LOGIC

def validate_features(data: dict) -> tuple[dict | None, str | None]:
    """
    Validate and coerce incoming feature dict.
    Returns (cleaned_data, error_message).
    """
    errors = []

    # Check all required features present
    missing = [f for f in FEATURE_COLS if f not in data]
    if missing:
        return None, f"Missing required features: {missing}"

    # Type + range validation
    try:
        lead_source = str(data["lead_source"]).upper()
        if lead_source not in VALID_LEAD_SOURCES:
            errors.append(f"lead_source '{lead_source}' not in {VALID_LEAD_SOURCES}")

        bv = str(data["business_vertical"]).upper()
        if bv not in VALID_VERTICALS:
            errors.append(f"business_vertical '{bv}' not in {VALID_VERTICALS}")

        hp = str(data["human_priority"]).upper()
        if hp not in VALID_PRIORITIES:
            errors.append(f"human_priority '{hp}' not in {VALID_PRIORITIES}")

        lead_value = float(data["lead_value"])
        if lead_value < 0:
            errors.append("lead_value must be >= 0")

        for col in ["days_in_pipeline", "time_in_current_stage",
                    "days_since_last_contact", "activity_count",
                    "task_count", "note_count"]:
            val = int(data[col])
            if val < 0:
                errors.append(f"{col} must be >= 0")

        stage_index = int(data["stage_index"])
        if stage_index not in [-1, 0, 1, 2, 3, 4, 5]:
            errors.append(f"stage_index must be in [-1, 0, 1, 2, 3, 4, 5]")

        stage_prob = float(data["stage_probability"])
        if not (0 <= stage_prob <= 100):
            errors.append("stage_probability must be 0-100")

        is_rotten = int(data["is_rotten"])
        if is_rotten not in [0, 1]:
            errors.append("is_rotten must be 0 or 1")

        has_task = int(data["has_upcoming_task"])
        if has_task not in [0, 1]:
            errors.append("has_upcoming_task must be 0 or 1")

    except (ValueError, TypeError) as e:
        return None, f"Type coercion error: {e}"

    if errors:
        return None, "; ".join(errors)

    # Return cleaned, normalised dict
    return {
        "lead_source":              lead_source,
        "business_vertical":        bv,
        "human_priority":           hp,
        "lead_value":               float(data["lead_value"]),
        "days_in_pipeline":         int(data["days_in_pipeline"]),
        "time_in_current_stage":    int(data["time_in_current_stage"]),
        "days_since_last_contact":  int(data["days_since_last_contact"]),
        "activity_count":           int(data["activity_count"]),
        "task_count":               int(data["task_count"]),
        "note_count":               int(data["note_count"]),
        "stage_index":              int(data["stage_index"]),
        "stage_probability":        float(data["stage_probability"]),
        "is_rotten":                int(data["is_rotten"]),
        "has_upcoming_task":        int(data["has_upcoming_task"]),
    }, None


def score_features(features: dict) -> dict:
    """
    Run the ML model on a validated feature dict.
    Returns scoring result ready to write back to MongoDB.
    """
    # Build DataFrame in the correct column order
    row = pd.DataFrame([features])

    # Encode categoricals using fitted LabelEncoders
    for col in CATEGORICAL_COLS:
        try:
            row[col] = ENCODERS[col].transform(row[col])
        except ValueError:
            # Unseen label — fall back to most common class (index 0)
            log.warning(f"Unseen label '{row[col].iloc[0]}' for {col}, using fallback")
            row[col] = 0

    # Enforce column order
    row = row[FEATURE_COLS]

    # Scale
    row_scaled = SCALER.transform(row)

    # Predict
    prob = float(MODEL.predict_proba(row_scaled)[0][1])
    ml_score = round(prob * 100, 1)

    # Map to priority tier
    if ml_score >= 65:
        ml_priority = "HIGH"
    elif ml_score >= 35:
        ml_priority = "MEDIUM"
    else:
        ml_priority = "LOW"

    # Top contributing features (for explainability UI).
    # Prefer importances persisted in feature_meta.json — the deployed model is a
    # CalibratedClassifierCV wrapper with no feature_importances_/coef_ attribute.
    if FEATURE_IMPORTANCES:
        importances = np.asarray(FEATURE_IMPORTANCES, dtype=float)
        feat_contributions = [
            {
                "feature": FEATURE_COLS[i],
                "importance": round(float(importances[i]), 4),
            }
            for i in np.argsort(importances)[::-1][:5]  # top 5
        ]
    else:
        # Fallback for a bare tree/linear model (no meta importances present)
        try:
            importances = MODEL.feature_importances_
            feat_contributions = [
                {
                    "feature": FEATURE_COLS[i],
                    "importance": round(float(importances[i]), 4),
                }
                for i in np.argsort(importances)[::-1][:5]
            ]
        except AttributeError:
            # LR fallback — use coefficient magnitudes
            coefs = np.abs(MODEL.coef_[0])
            feat_contributions = [
                {
                    "feature": FEATURE_COLS[i],
                    "importance": round(float(coefs[i]), 4),
                }
                for i in np.argsort(coefs)[::-1][:5]
            ]

    return {
        "mlScore":               ml_score,
        "mlPriority":            ml_priority,
        "conversionProbability": round(prob, 4),
        "model":                 MODEL_NAME,
        "topFeatures":           feat_contributions,
        "scoredAt":              datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok" if ARTIFACTS_LOADED else "degraded",
        "model":           MODEL_NAME if ARTIFACTS_LOADED else None,
        "features":        len(FEATURE_COLS) if ARTIFACTS_LOADED else 0,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }), 200 if ARTIFACTS_LOADED else 503


@app.route("/features", methods=["GET"])
def features():
    """Returns the expected feature schema — useful for Node.js validation."""
    if not ARTIFACTS_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    schema = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            if col == "lead_source":
                schema[col] = {"type": "string", "enum": VALID_LEAD_SOURCES}
            elif col == "business_vertical":
                schema[col] = {"type": "string", "enum": VALID_VERTICALS}
            elif col == "human_priority":
                schema[col] = {"type": "string", "enum": VALID_PRIORITIES}
        elif col in ["is_rotten", "has_upcoming_task"]:
            schema[col] = {"type": "integer", "enum": [0, 1]}
        elif col in ["stage_index"]:
            schema[col] = {"type": "integer", "enum": [-1, 0, 1, 2, 3, 4, 5]}
        elif col in ["stage_probability"]:
            schema[col] = {"type": "float", "min": 0, "max": 100}
        else:
            schema[col] = {"type": "float", "min": 0}

    return jsonify({
        "feature_cols":      FEATURE_COLS,
        "categorical_cols":  CATEGORICAL_COLS,
        "schema":            schema,
        "model":             MODEL_NAME,
    }), 200


# ── 404 handler
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") == "development"
    log.info(f"Starting LeadXpert ML service on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
