# levelDetection/service/levelDetection.py
import re
from typing import Dict, Any, List
from .ollama_client import generate_json, mistral_generate

# Minimal schema: ONLY what you want back
_MIN_SCHEMA = {
    "type": "object",
    "properties": {
        "total_score": {"type": "integer", "minimum": 0, "maximum": 27},
        "level": {
            "type": "string",
            "enum": ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"],
        },
    },
    "required": ["total_score", "level"],
    "additionalProperties": False,
}

# Fallback regex for your finetune text format:
# PHQ-9 Score: 26
# Depression Level: Severe
_SCORE_RE = re.compile(r"PHQ-9\s*Score\s*:\s*(\d{1,2})", re.IGNORECASE)
_LEVEL_RE = re.compile(
    r"Depression\s*Level\s*:\s*(Minimal|Mild|Moderate|Moderately\s+Severe|Severe)",
    re.IGNORECASE,
)

def _parse_text_fallback(text: str) -> Dict[str, Any]:
    score = None
    level = None

    m1 = __SCORE_RE.search(text or "")
    if m1:
        try:
            score = int(m1.group(1))
        except Exception:
            pass

    m2 = _LEVEL_RE.search(text or "")
    if m2:
        level = m2.group(1).title().replace("Severe", "Severe")\
                 .replace("Moderately severe", "Moderately Severe")

    if score is None or level is None:
        raise ValueError("Could not parse score/level from text fallback.")
    return {"phq9_score": score, "level": level}

def detect_from_phq9_answers(phq9_answers: List[str]) -> Dict[str, Any]:
    """
    Send free-text PHQ-9 answers to the fine-tuned model.
    Expect ONLY total_score and level. If JSON fails, parse text fallback.
    """
    # Numbered list helps the model keep order
    numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(phq9_answers or []))

    # Primary: strict JSON
    user_prompt = f"""
Analyze the following PHQ-9 responses and output ONLY JSON with:
- "total_score": integer 0..27
- "level": one of ["Minimal","Mild","Moderate","Moderately Severe","Severe"]

PHQ-9 responses (in order):
{numbered}
"""
    try:
        data = generate_json(user_prompt, schema=_MIN_SCHEMA)
        return {"phq9_score": data["total_score"], "level": data["level"]}
    except Exception:
        txt_prompt = f"""
[INST] Analyze the following PHQ-9 responses and provide the score and depression level.

{numbered} [/INST]
Respond EXACTLY in this format (no extra words):
PHQ-9 Score: <number>
Depression Level: <Minimal|Mild|Moderate|Moderately Severe|Severe>
"""
        text = mistral_generate(txt_prompt, temperature=0.0, num_predict=120)
        return _parse_text_fallback(text)
