import numpy as np
import pandas as pd
import random
import argparse
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# NOTE: these enums MUST match the backend schema in
# leadXpert_server/src/types/shared.types.ts exactly. The Flask validator
# (app.py) rejects any value it hasn't seen during training, so keep them
# in sync with LeadSource / BusinessVertical / LeadPriority.
LEAD_SOURCES = [
    "FACEBOOK", "INSTAGRAM", "TIKTOK", "VIBER", "WHATSAPP",
    "WALK_IN", "REFERRAL", "GOOGLE", "WEBSITE", "YOUTUBE",
    "EMAIL", "PHONE_CALL", "LINKEDIN", "EVENT", "COLD_OUTREACH", "OTHER",
]

BUSINESS_VERTICALS = [
    "EDUCATION_CONSULTANCY", "DIGITAL_MARKETING", "IT_SOFTWARE",
    "LEGAL_FINANCIAL", "REAL_ESTATE", "RECRUITMENT",
    "EVENT_MANAGEMENT", "GENERAL",
]

# Human priority — matches LeadPriority (includes URGENT)
PRIORITY_LEVELS = ["URGENT", "HIGH", "MEDIUM", "LOW"]

# Base conversion rates per source (calibrated for ~22% overall).
# Warm/inbound channels (referral, walk-in, direct messaging) convert higher;
# broad social and cold outreach convert lower.
SOURCE_CONVERSION_WEIGHTS = {
    "FACEBOOK":       0.14,
    "INSTAGRAM":      0.15,
    "TIKTOK":         0.11,
    "VIBER":          0.24,
    "WHATSAPP":       0.26,
    "WALK_IN":        0.34,
    "REFERRAL":       0.42,
    "GOOGLE":         0.20,
    "WEBSITE":        0.22,
    "YOUTUBE":        0.13,
    "EMAIL":          0.18,
    "PHONE_CALL":     0.25,
    "LINKEDIN":       0.20,
    "EVENT":          0.28,
    "COLD_OUTREACH":  0.08,
    "OTHER":          0.12,
}

# Channel mix for a Kathmandu SME (social-heavy). Sums to 1.0, aligned by index
# with LEAD_SOURCES above.
SOURCE_DISTRIBUTION = [
    0.15, 0.13, 0.07, 0.06, 0.08,   # FACEBOOK INSTAGRAM TIKTOK VIBER WHATSAPP
    0.05, 0.10, 0.08, 0.07, 0.03,   # WALK_IN REFERRAL GOOGLE WEBSITE YOUTUBE
    0.04, 0.05, 0.03, 0.03, 0.02, 0.01,  # EMAIL PHONE_CALL LINKEDIN EVENT COLD_OUTREACH OTHER
]
# Aligned by index with BUSINESS_VERTICALS. Sums to 1.0.
VERTICAL_DISTRIBUTION = [
    0.20, 0.22, 0.15, 0.10, 0.08, 0.08, 0.07, 0.10,
]

# Open pipeline stages only (0-4) — leads being SCORED are always open.
# NEW stage probability is 20 to match the live PipelineStage schema default
# (pipeline-stage.model.ts), which is the source of truth for production data.
OPEN_STAGES = [
    {"name": "NEW",         "index": 0, "probability": 20},
    {"name": "CONTACTED",   "index": 1, "probability": 20},
    {"name": "QUALIFIED",   "index": 2, "probability": 40},
    {"name": "PROPOSAL",    "index": 3, "probability": 60},
    {"name": "NEGOTIATION", "index": 4, "probability": 80},
]

# Fresh (day-one) lead coverage is now produced by the main generator itself:
# days_in_pipeline starts at 0 and interaction counts are age-scaled (see
# generate_lead), so all-zero-history leads occur naturally AND connect smoothly
# to aged leads. An explicit all-zero *spike* at exactly day 0 is deliberately
# NOT used: piling tens of thousands of zero-activity leads onto a single day
# made the model treat "days_in_pipeline == 0" as a special low-conversion
# marker, so a lead jumped ~20 points the instant it aged one day. Keeping the
# continuum smooth (fraction 0) is what actually fulfils "the model must have
# seen fresh leads" without the boundary artifact.
FRESH_FRACTION = 0.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def generate_lead(lead_id: int, fresh: bool = False) -> dict:
    source   = random.choices(LEAD_SOURCES, weights=SOURCE_DISTRIBUTION, k=1)[0]
    vertical = random.choices(BUSINESS_VERTICALS, weights=VERTICAL_DISTRIBUTION, k=1)[0]

    base_conv = SOURCE_CONVERSION_WEIGHTS[source]

    # Lead value (NPR) per vertical
    value_ranges = {
        "EDUCATION_CONSULTANCY": (15000, 400000),
        "DIGITAL_MARKETING":     (10000, 300000),
        "IT_SOFTWARE":           (20000, 800000),
        "LEGAL_FINANCIAL":       (10000, 500000),
        "REAL_ESTATE":           (50000, 800000),
        "RECRUITMENT":           (8000,  200000),
        "EVENT_MANAGEMENT":      (15000, 600000),
        "GENERAL":               (5000,  200000),
    }
    v_min, v_max = value_ranges[vertical]
    lead_value = round(np.random.lognormal(
        mean=np.log((v_min + v_max) / 3), sigma=0.7
    ))
    lead_value   = clamp(lead_value, v_min, v_max)
    value_factor = clamp((lead_value - v_min) / (v_max - v_min), 0, 1)

    if fresh:
        # Day-one lead: exactly the all-zero vector the live system sends the
        # moment a lead is created (see extractFeatures in scoring.service.ts —
        # never-contacted falls back to days_in_pipeline, which is also 0 here).
        days_in_pipeline        = 0
        time_in_current_stage   = 0
        days_since_last_contact = 0
        has_upcoming_task       = False
        activity_count          = 0
        task_count              = 0
        note_count              = 0
        is_rotten               = False
        stage                   = OPEN_STAGES[0]  # NEW stage
    else:
        # Timeline — days_in_pipeline starts at 0 so the "just created" state is
        # part of the main population, not an isolated island. time_in_current
        # stage can never exceed the lead's total age.
        days_in_pipeline        = random.randint(0, 180)
        time_in_current_stage   = random.randint(0, days_in_pipeline)
        days_since_last_contact = random.randint(0, min(60, days_in_pipeline))
        has_upcoming_task       = random.random() < (0.40 + 0.25 * value_factor)

        # Interaction counts accrue with pipeline age. Same-day logging is
        # possible (a lead created today can already have a note or two), but a
        # 1-day-old lead should NOT carry the same ~5 interactions as a
        # 6-month-old one. Tying counts to age removes the hard gap the live app
        # fell into: previously activity was independent of age, so young leads
        # jumped straight to ~5 interactions and the score teleported to ~80 the
        # moment a brand-new lead logged its first note/task. age_factor ramps
        # from 0.125 on day 0 to ~1.0 for a mature lead.
        age_factor     = (days_in_pipeline + 1) / (days_in_pipeline + 8)
        activity_count = max(0, int(np.random.poisson((2 + 4 * value_factor) * age_factor)))
        task_count     = max(0, int(activity_count * random.uniform(0.3, 0.6)))
        note_count     = max(0, int(activity_count * random.uniform(0.2, 0.45)))

        is_rotten = days_since_last_contact > 14 and random.random() < 0.45

        # Current pipeline stage (OPEN stages 0-4 only)
        # Higher-stage leads are more likely to be in advanced stages if active
        if is_rotten:
            stage_weights = [0.40, 0.28, 0.18, 0.09, 0.05]
        else:
            stage_weights = [0.28, 0.25, 0.22, 0.15, 0.10]

        stage = random.choices(OPEN_STAGES, weights=stage_weights, k=1)[0]

    stage_index       = stage["index"]
    stage_probability = stage["probability"]

    # Stage boost: leads further in pipeline have already proven more intent
    stage_boost = stage_index * 0.04  # 0 to 0.16

    # Conversion probability (all signals combined)
    interaction_boost = clamp(activity_count / 25, 0, 0.18)
    value_boost       = clamp(value_factor * 0.10, 0, 0.10)
    rotten_penalty    = -0.12 if is_rotten else 0
    contact_penalty   = -0.08 if days_since_last_contact > 7 else 0
    task_boost_val    = 0.06 if has_upcoming_task else 0
    recency_penalty   = -0.05 if days_in_pipeline > 90 else 0

    conversion_prob = clamp(
        base_conv + stage_boost + interaction_boost + value_boost
        + rotten_penalty + contact_penalty + task_boost_val + recency_penalty,
        0.02, 0.80
    )

    is_converted = int(random.random() < conversion_prob)

    # Human priority (biased heuristic — feature showing cognitive bias).
    # Weights align by index with PRIORITY_LEVELS = [URGENT, HIGH, MEDIUM, LOW].
    if source == "REFERRAL" or lead_value > v_max * 0.6:
        human_priority = random.choices(PRIORITY_LEVELS, weights=[0.15, 0.45, 0.30, 0.10])[0]
    elif is_rotten or days_since_last_contact > 20:
        human_priority = random.choices(PRIORITY_LEVELS, weights=[0.03, 0.10, 0.30, 0.57])[0]
    elif activity_count > 8:
        human_priority = random.choices(PRIORITY_LEVELS, weights=[0.20, 0.40, 0.30, 0.10])[0]
    else:
        human_priority = random.choices(PRIORITY_LEVELS, weights=[0.05, 0.18, 0.52, 0.25])[0]

    return {
        "lead_id":                  lead_id,
        "lead_source":              source,
        "business_vertical":        vertical,
        "human_priority":           human_priority,
        "lead_value":               lead_value,
        "days_in_pipeline":         days_in_pipeline,
        "time_in_current_stage":    time_in_current_stage,
        "days_since_last_contact":  days_since_last_contact,
        "activity_count":           activity_count,
        "task_count":               task_count,
        "note_count":               note_count,
        "stage_index":              stage_index,       # 0-4 only
        "stage_probability":        stage_probability, # 10-80 only
        "is_rotten":                int(is_rotten),
        "has_upcoming_task":        int(has_upcoming_task),
        "converted":                is_converted,
    }

def generate_dataset(n: int, output_path: str, chunk_size: int = 50_000):
    print(f"\n{'='*55}")
    print(f"  LeadXpert Synthetic Dataset Generator v2")
    print(f"{'='*55}")
    print(f"  Records : {n:,}  |  Output : {output_path}")
    print(f"{'='*55}\n")

    total_chunks  = (n + chunk_size - 1) // chunk_size
    generated     = 0
    conv_count    = 0

    for ci in range(total_chunks):
        start   = ci * chunk_size
        end     = min(start + chunk_size, n)
        records = [
            generate_lead(start + i, fresh=(random.random() < FRESH_FRACTION))
            for i in range(end - start)
        ]
        df      = pd.DataFrame(records)

        conv_count += df["converted"].sum()
        generated  += len(df)

        df.to_csv(output_path, mode="w" if ci == 0 else "a",
                  index=False, header=(ci == 0))

        pct  = generated / n * 100
        conv = conv_count / generated * 100
        bar  = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        print(f"  [{bar}] {pct:5.1f}%  {generated:>9,}  conv: {conv:.1f}%")

    mb = os.path.getsize(output_path) / 1024**2
    print(f"\n  ✓ {generated:,} records → {output_path} ({mb:.1f} MB)")
    print(f"  Conversion rate : {conv_count/generated*100:.2f}%")
    print(f"  Class 1         : {int(conv_count):,}")
    print(f"  Class 0         : {int(generated-conv_count):,}")
    print(f"  Imbalance ratio : 1 : {(generated-conv_count)/conv_count:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=750_000)
    parser.add_argument("--out", type=str, default="leadxpert_leads.csv")
    args = parser.parse_args()
    generate_dataset(n=args.n, output_path=args.out)
