from pydantic import BaseModel
from typing import List


class VisualAnalysis(BaseModel):
    description: str
    objects: List[str]
    visible_text: List[str]
    important_details: List[str]


class VisualResponse(BaseModel):
    status: str
    message: str
    result: VisualAnalysis