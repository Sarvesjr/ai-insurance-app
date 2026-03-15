def estimate_cost(severity):

    if severity == "Minor Dent":
        return "Estimated Repair Cost: $100 - $500"

    elif severity == "Moderate Damage":
        return "Estimated Repair Cost: $500 - $2500"

    elif severity == "Severe Accident":
        return "Estimated Repair Cost: $2500 - $10000"

    else:
        return "Unable to estimate cost"