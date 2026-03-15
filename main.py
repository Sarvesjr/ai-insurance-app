from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from image_analyzer import analyze_damage
from cost_estimator import estimate_payout
import time

app = FastAPI(title="InsureAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/claim_model.pkl")

@app.get("/")
def home():
    return {"message": "InsureAI API is running!"}

@app.post("/assess-claim")
async def assess_claim(
    name: str = Form(...),
    age: int = Form(...),
    gender: int = Form(...),
    driving_license: int = Form(...),
    region_code: float = Form(...),
    previously_insured: int = Form(...),
    vehicle_age: int = Form(...),
    vehicle_damage: int = Form(...),
    annual_premium: float = Form(...),
    policy_sales_channel: float = Form(...),
    vintage: int = Form(...),
    damage_image: UploadFile = File(...)
):
    start_time = time.time()

    features = np.array([[
        gender, age, driving_license, region_code,
        previously_insured, vehicle_age, vehicle_damage,
        annual_premium, policy_sales_channel, vintage
    ]])
    claim_result = int(model.predict(features)[0])
    claim_probability = float(model.predict_proba(features)[0][1])

    image_bytes = await damage_image.read()
    damage_analysis = analyze_damage(image_bytes)

    payout = None
    if claim_result == 1:
        payout = estimate_payout(
            annual_premium=annual_premium,
            damage_percent=damage_analysis["damage_percent"],
            severity=damage_analysis["severity"],
            claim_probability=claim_probability
        )

    processing_time = round(time.time() - start_time, 1)

    return {
        "name": name,
        "claim_approved": bool(claim_result),
        "claim_probability": round(claim_probability * 100, 1),
        "verdict": "CLAIM APPROVED" if claim_result == 1 else "CLAIM REJECTED",
        "damage_analysis": damage_analysis,
        "payout": payout,
        "processing_time": processing_time
    }