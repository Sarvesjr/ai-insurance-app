def estimate_payout(annual_premium, damage_percent, severity, claim_probability):
    insured_value = annual_premium * 8
    multipliers = {"MINOR": 0.3, "MODERATE": 0.6, "SEVERE": 0.9}
    multiplier = multipliers.get(severity, 0.5)
    raw_payout = insured_value * (damage_percent / 100) * multiplier
    adjusted_payout = raw_payout * claim_probability
    deductible = adjusted_payout * 0.05
    final_payout = adjusted_payout - deductible
    return {
        "insured_value": round(insured_value, 2),
        "raw_payout": round(raw_payout, 2),
        "deductible": round(deductible, 2),
        "final_payout": round(final_payout, 2),
        "payout_percentage": round((final_payout / insured_value) * 100, 1)
    }