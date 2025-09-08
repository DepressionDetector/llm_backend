# levelDetection/service/ollama_client.py
import os
import json
import requests
from jsonschema import validate  # pip install jsonschema

# Ngrok → Ollama host (override via env)
BASE = os.getenv("OLLAMA_BASE", "https://55713976f485.ngrok-free.app").rstrip("/")

LEVEL_MODEL = os.getenv("LEVEL_MODEL", "mistral-LevelDetector")
AUTH = None 

STOPS = ["</s>", "[INST]", "User:", "\nUser"]


def _post_generate(payload: dict, timeout: int = 120) -> dict:
    """Low-level wrapper for /api/generate with basic error surfacing."""
    r = requests.post(
        f"{BASE}/api/generate",
        auth=AUTH,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def mistral_generate(prompt: str, temperature: float = 0.3, num_predict: int = 120) -> str:
    """Plain text generation (no JSON)."""
    data = _post_generate(
        {
            "model": "mistral-LevelDetector",         
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": num_predict,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": STOPS,
            },
        }
    )
    return (data.get("response") or "").strip()


def generate_json(user_prompt: str, schema: dict, num_predict: int = 200) -> dict:
    """
    Ask the model to return STRICT JSON validated against `schema`.
    Mirrors your working pattern (format='json' + jsonschema.validate).
    """
    instruct = (
        "Return strictly valid JSON only. No explanations.\n"
        f"Match this schema as closely as possible:\n{json.dumps(schema)}\n\n"
        f"{user_prompt}\n"
    )

    data = _post_generate(
        {
            "model": "mistral-LevelDetector",         
            "prompt": instruct,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": 4096,
                "num_predict": num_predict,
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": STOPS,
            },
        }
    )

    obj = json.loads(data.get("response") or "{}")
    validate(instance=obj, schema=schema)
    return obj
