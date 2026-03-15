import json

def handler(request):
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "AI Insurance API running successfully"
        })
    }