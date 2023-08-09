from pydantic import BaseModel


class AssistantDoResponse(BaseModel):
    query: str
