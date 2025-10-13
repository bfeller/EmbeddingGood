from __future__ import annotations

from enum import Enum
from typing import List, Union

from pydantic import BaseModel, Field, field_validator


class EmbedType(str, Enum):
    query = "query"
    document = "document"


class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    type: EmbedType = Field(default=EmbedType.document)
    dimension: int = Field(default=768, description="Embedding size: 128|256|512|768")

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v not in (128, 256, 512, 768):
            raise ValueError("dimension must be one of 128, 256, 512, 768")
        return v


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model: str


