import json
from image_analyzer import analyze_damage
from cost_estimator import estimate_cost

def handler(request):

    # Example test image
    image_path = "test_car.jpg"

    result = analyze_damage(image_path)

    cost = estimate_cost(result["severity"])

    response = {
        "severity": result["severity"],
        "damage_ratio": result["damage_ratio"],
        "cost_estimate": cost
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }