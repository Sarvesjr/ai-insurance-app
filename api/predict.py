"""
api/predict.py  —  Vercel Serverless Function
Runs claim prediction and payout estimation.
POST /api/predict
Receives form data + pre-computed damage numbers from /api/analyze
"""

from http.server import BaseHTTPRequestHandler
import json, math, time


# ── LOGISTIC REGRESSION (pure Python, no sklearn, no .pkl) ───────────────────
INTERCEPT = -1.872
COEF = {
    "gender":              0.089,
    "age":                 0.021,
    "driving_license":     0.412,
    "region_code":        -0.003,
    "previously_insured": -2.105,
    "vehicle_age":         0.634,
    "vehicle_damage":      2.187,
    "annual_premium":      0.000012,
    "policy_sales_channel":-0.002,
    "vintage":             0.0008,
}

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def predict_claim(features: dict):
    logit = INTERCEPT + sum(COEF[k] * float(features.get(k, 0)) for k in COEF)
    prob  = _sigmoid(logit)
    return (1 if prob >= 0.5 else 0), round(prob, 4)


# ── PAYOUT ESTIMATOR ─────────────────────────────────────────────────────────
SEVERITY_MULTIPLIER = {"MINOR": 0.3, "MODERATE": 0.6, "SEVERE": 0.9}
VEHICLE_AGE_FACTOR  = {"< 1 Year": 1.35, "1-2 Years": 1.15, "> 2 Years": 0.90}
CHANNEL_FACTOR      = {"Agent": 1.05, "Direct": 1.00, "Broker": 1.08, "Online": 0.95}

def estimate_payout(annual_premium, damage_percent, severity,
                    claim_probability, vehicle_age, channel,
                    prev_insured, age, vintage):
    insured_value   = annual_premium * 8
    base_payout     = insured_value * (damage_percent / 100) * SEVERITY_MULTIPLIER.get(severity, 0.5)
    adjusted_payout = base_payout * claim_probability
    adjusted_payout *= VEHICLE_AGE_FACTOR.get(vehicle_age, 1.0)
    adjusted_payout *= CHANNEL_FACTOR.get(channel, 1.0)
    adjusted_payout *= (0.88 if prev_insured == "Yes" else 1.0)
    adjusted_payout *= (1.0 + max(0, (age - 25)) * 0.003)
    adjusted_payout *= (1.0 + min(vintage / 1000, 0.15))
    deductible   = adjusted_payout * 0.05
    final_payout = adjusted_payout - deductible
    return {
        "insured_value":     round(insured_value, 2),
        "raw_payout":        round(adjusted_payout, 2),
        "deductible":        round(deductible, 2),
        "final_payout":      round(final_payout, 2),
        "payout_percentage": round((final_payout / insured_value) * 100, 1) if insured_value else 0,
    }


# ── VERCEL HANDLER ────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def log_message(self, *a): pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        t0 = time.time()
        try:
            length  = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except Exception as e:
            return self._err(400, f"Bad JSON: {e}")

        vehicle_age_raw = payload.get("vehicleAge", "> 2 Years")
        prev_insured    = payload.get("prevInsured", "No")
        damage_hist     = payload.get("damageHistory", "No")
        channel         = payload.get("channel", "Direct")
        age             = int(payload.get("age", 30))
        vintage         = int(payload.get("vintage", 100))
        annual_premium  = float(payload.get("annualPremium", 30000))

        features = {
            "gender":              1 if payload.get("gender", "Male") == "Male" else 0,
            "age":                 age,
            "driving_license":     1 if payload.get("licenceValid", "Yes") == "Yes" else 0,
            "region_code":         float(payload.get("regionCode", 28) or 28),
            "previously_insured":  1 if prev_insured == "Yes" else 0,
            "vehicle_age":         {"< 1 Year": 0, "1-2 Years": 1, "> 2 Years": 2}.get(vehicle_age_raw, 2),
            "vehicle_damage":      1 if damage_hist == "Yes" else 0,
            "annual_premium":      annual_premium,
            "policy_sales_channel":{"Agent": 26, "Direct": 152, "Broker": 124, "Online": 160}.get(channel, 152),
            "vintage":             vintage,
        }

        claim_result, claim_prob = predict_claim(features)

        damage_percent = float(payload.get("damagePercent", 45 if damage_hist == "Yes" else 15))
        severity       = str(payload.get("severity", "MODERATE" if damage_hist == "Yes" else "MINOR"))
        ai_confidence  = float(payload.get("aiConfidence", 0.75))
        description    = str(payload.get("description", "Estimated from policy data."))

        payout = estimate_payout(
            annual_premium=annual_premium, damage_percent=damage_percent,
            severity=severity, claim_probability=claim_prob,
            vehicle_age=vehicle_age_raw, channel=channel,
            prev_insured=prev_insured, age=age, vintage=vintage,
        ) if claim_result == 1 else {
            "insured_value": round(annual_premium * 8, 2),
            "raw_payout": 0, "deductible": 0, "final_payout": 0, "payout_percentage": 0,
        }

        self._ok({
            "claim_approved":    bool(claim_result),
            "verdict":           "CLAIM APPROVED" if claim_result == 1 else "CLAIM REJECTED",
            "claim_probability": round(claim_prob * 100, 1),
            "confidence":        ai_confidence,
            "damage_score":      round(damage_percent / 100, 3),
            "damage_percent":    damage_percent,
            "severity":          severity,
            "description":       description,
            "raw_payout":        payout["raw_payout"],
            "insured_value":     payout["insured_value"],
            "final_payout":      payout["final_payout"],
            "deductible":        payout["deductible"],
            "elapsed_sec":       round(time.time() - t0, 3),
        })

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _ok(self, body: dict):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _err(self, status: int, msg: str):
        data = json.dumps({"error": msg}).encode()
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)