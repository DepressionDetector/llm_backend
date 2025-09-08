# levelDetection/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal

from ..service.levelDetection import detect_from_phq9_answers

router = APIRouter()

PHQ9Level = Literal["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]

class DetectFromPHQ9Request(BaseModel):
    phq9Answers: List[str] = Field(default_factory=list, description="User free-text answers in PHQ-9 order")

class DetectFromPHQ9Response(BaseModel):
    phq9_score: int
    level: PHQ9Level

@router.post("/detect", response_model=DetectFromPHQ9Response)
async def detect_from_phq9(req: DetectFromPHQ9Request):
    try:
        result = detect_from_phq9_answers(req.phq9Answers or [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DetectFromPHQ9Response(
        phq9_score=result["phq9_score"],
        level=result["level"],
    )
