import numpy as np
from bson.objectid import ObjectId
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Any, ClassVar


class TextAndRoles(BaseModel):
    text: str
    roles: list[str]
    allowed_roles: ClassVar[set] = set()

    @field_validator("roles")
    def check_roles(cls, v):
        if not isinstance(v, list):
            raise TypeError(
                f"'roles' must be a list of strings, got {type(v).__name__}"
            )

        lowercased_roles = [r.lower() for r in v]
        for r in lowercased_roles:
            if not isinstance(r, str):
                raise TypeError(
                    f"All items in 'roles' must be strings, got {type(r).__name__} for item {r}"
                )
            if r not in cls.allowed_roles:
                raise ValueError(
                    f"Role '{r}' is not an allowed role. Allowed roles: {cls.allowed_roles}"
                )

        return lowercased_roles

    @classmethod
    def with_roles(cls, text: str, roles: list[str], allowed_roles: set):
        cls.allowed_roles = allowed_roles
        return cls(text=text, roles=roles)


class NumpyArray(BaseModel):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, np.ndarray):
            return v
        raise ValueError("Invalid type")


class ToUpload(BaseModel):
    text: str
    embedding: NumpyArray | list
    roles: list[str]


# Example RetrievedObject:
# {
#     "_id": ObjectId("66cb941cc09ea7c6d1676ecb"),
#     "text": "Go easy on the kid",
#     "roles": ["ceo"],
#     "search_score": -0.011121630668640137,
# }
class RetrievedObject(BaseModel):
    _id: ObjectId
    roles: list[str]
    text: str
    search_score: float
    extra_field: dict[str, Any] = {}

    model_config = ConfigDict(
        extra="allow",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.extra_field = {k: v for k, v in data.items() if k not in self.model_fields}