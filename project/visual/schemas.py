from pydantic import BaseModel


class VisualResponse(BaseModel):
    status: str
    message: str
    result: str