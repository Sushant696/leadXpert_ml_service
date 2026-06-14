# LeadXpert ML Microservice

Python Flask service that serves the trained Random Forest lead scoring model.

## Directory Structure

```
ml-service/
├── app.py                  ← Flask app (all endpoints)
├── requirements.txt        ← Python dependencies
├── ecosystem.config.js     ← PM2 config for deployment
├── models/
│   ├── best_model.pkl      ← Trained Random Forest
│   ├── scaler.pkl          ← StandardScaler
│   ├── encoders.pkl        ← LabelEncoders for categoricals
│   └── feature_meta.json   ← Feature list + model name
└── README.md
```

## Local Dev Setup

```bash
cd ml-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run dev server (port 5001)
python app.py
```

Service starts at `http://localhost:5001`

---

## Endpoints

### GET /health
Liveness check. Returns 200 if model is loaded, 503 if not.

```json
{ "status": "ok", "model": "Random Forest", "features": 14 }
```

### GET /features
Returns the expected feature schema for validation.

### POST /score
Score a single lead. All 14 features required.

**Request:**
```json
{
  "lead_source": "REFERRAL",
  "business_vertical": "CONSULTING",
  "human_priority": "HIGH",
  "lead_value": 250000,
  "days_in_pipeline": 15,
  "time_in_current_stage": 5,
  "days_since_last_contact": 2,
  "activity_count": 10,
  "task_count": 4,
  "note_count": 3,
  "stage_index": 3,
  "stage_probability": 60,
  "is_rotten": 0,
  "has_upcoming_task": 1
}
```

**Response:**
```json
{
  "mlScore": 73.6,
  "mlPriority": "HIGH",
  "conversionProbability": 0.736,
  "model": "Random Forest",
  "topFeatures": [
    { "feature": "lead_source", "importance": 0.2814 },
    { "feature": "activity_count", "importance": 0.1571 }
  ],
  "scoredAt": "2026-06-27T10:00:00Z"
}
```

**Score → Priority mapping:**
| mlScore | mlPriority |
|---|---|
| 65–100 | HIGH |
| 35–64  | MEDIUM |
| 0–34   | LOW |

### POST /score/batch
Score up to 500 leads at once. Body: `{ "leads": [ { "leadId": "...", ...features } ] }`

---

## VPS Deployment (Hostinger)

```bash
# 1. Upload ml-service/ to /var/www/leadxpert/ml-service/
# 2. Create venv and install
cd /var/www/leadxpert/ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Start with PM2
pm2 start ecosystem.config.js --only ml-service
pm2 save
pm2 startup   # follow the printed command to persist on reboot

# 4. Verify
curl http://localhost:5001/health
```

**Important:** The Flask service binds to `0.0.0.0:5001` but should NOT be exposed externally.
In Nginx, do NOT proxy_pass port 5001. Only the Node.js API (port 5500) is public-facing.

```nginx
# ✅ Public — proxied through Nginx
location /api { proxy_pass http://localhost:5500; }

# ❌ Do NOT add this — Flask stays internal
# location /ml { proxy_pass http://localhost:5001; }
```

---

## Node.js Integration

Copy these files into your backend:

```
src/
├── services/ml/
│   └── scoring.service.ts     ← core scoring logic + Flask call
├── hooks/
│   └── lead.scoring.hook.ts   ← Mongoose post-save hook
└── routes/
    └── scoring.router.ts      ← /api/ml/* endpoints
```

**Register the router in your Express app:**
```typescript
import { scoringRouter } from "@/routes/scoring.router";
app.use("/api/ml", scoringRouter);
```

**Set the env variable:**
```env
ML_SERVICE_URL=http://localhost:5001
```

**The hook fires automatically** after every lead save/update that touches
`stageId`, `activityCount`, `taskCount`, `noteCount`, `isRotten`, etc.
It's fire-and-forget — never blocks the API response.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5001` | Flask port |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `ML_SERVICE_URL` | `http://localhost:5001` | Used by Node.js backend |
