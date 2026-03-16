"""
api/analyze.py  —  Vercel Serverless Function
Calls OpenRouter vision API to detect car and analyse damage.
POST /api/analyze
Body: { "imageBase64": "data:image/jpeg;base64,..." }
"""

from http.server import BaseHTTPRequestHandler
import json, os, base64, io, urllib.request, urllib.error, traceback


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct:free"

USER_PROMPT = """Look at this image carefully.

First: does this image contain a car or any vehicle?

If NO car or vehicle is present, respond with ONLY this JSON:
{"car_detected": false, "reason": "describe what you see instead"}

If a car IS present, assess all visible damage and respond with ONLY this JSON:
{
  "car_detected": true,
  "damage_percent": <integer 0-95>,
  "severity": "<MINOR|MODERATE|SEVERE>",
  "confidence": <float 0.0-1.0>,
  "description": "<one sentence describing the damage>",
  "damaged_parts": ["e.g. front bumper", "hood"]
}

Severity guide:
- MINOR    = light scratches or small dents, damage_percent < 25
- MODERATE = visible panel damage or cracks, damage_percent 25-60
- SEVERE   = structural deformation or major crush, damage_percent > 60

Reply with ONLY the JSON object. No markdown, no explanation, no extra text."""


def compress_image(b64_string: str) -> str:
    try:
        from PIL import Image
        raw = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        w, h = img.size
        MAX = 512
        if w > MAX or h > MAX:
            if w > h: h = round(h * MAX / w); w = MAX
            else:     w = round(w * MAX / h); h = MAX
            img = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        raw = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
        return raw


class handler(BaseHTTPRequestHandler):

    def log_message(self, *a): pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            self._handle_post()
        except Exception:
            self._err(500, traceback.format_exc())

    def _handle_post(self):
        # 1. Parse body
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))

        img_b64 = payload.get("imageBase64", "")
        if not img_b64:
            return self._err(400, "imageBase64 is required")

        # 2. Get API key
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return self._err(500, "OPENROUTER_API_KEY is not set in Vercel environment variables")

        # 3. Compress image
        compressed = compress_image(img_b64)

        # 4. Build OpenRouter request
        or_payload = json.dumps({
            "model": OPENROUTER_MODEL,
            "max_tokens": 512,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{compressed}"
                            }
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT
                        }
                    ]
                }
            ]
        }).encode("utf-8")

        # 5. Call OpenRouter
        req = urllib.request.Request(
            OPENROUTER_API_URL,
            data=or_payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer":  "https://ai-insurance-predictor-app.vercel.app",
                "X-Title":       "InsureAI",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                or_data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return self._err(502, f"OpenRouter API error {e.code}: {body}")
        except Exception as e:
            return self._err(502, f"OpenRouter request failed: {str(e)}")

        # 6. Parse response
        try:
            text = or_data["choices"][0]["message"]["content"].strip()
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
        except Exception as e:
            return self._err(502, f"Could not parse AI response: {str(e)}")

        # 7. No car detected
        if not result.get("car_detected", False):
            reason = result.get("reason", "No car found in image")
            return self._err(422, f"NO_CAR_DETECTED: {reason}")

        # 8. Return damage analysis
        self._ok({
            "car_detected":   True,
            "damage_percent": int(result.get("damage_percent", 50)),
            "severity":       str(result.get("severity", "MODERATE")),
            "confidence":     float(result.get("confidence", 0.85)),
            "description":    str(result.get("description", "")),
            "damaged_parts":  result.get("damaged_parts", []),
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