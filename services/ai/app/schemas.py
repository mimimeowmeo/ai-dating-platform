from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_BODY_BYTES = MAX_BASE64_LENGTH + 4096


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    imageBase64: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH)
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]
    requestId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["verified", "rejected", "unavailable"]
    reasonCode: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z][A-Z0-9_]*$")
    modelName: str | None = Field(default=None, min_length=1, max_length=128)
    modelVersion: str | None = Field(default=None, min_length=1, max_length=128)
    livenessScore: Score | None = None
    faceMatchScore: Score | None = None


class ProviderResult(VerificationResult):
    """Provider 的決策須明確表明活體與身分比對；本服務不自行訂門檻。"""

    verificationType: Literal["identity_verification"]
    livenessVerified: bool
    identityVerified: bool

    @model_validator(mode="after")
    def require_verified_evidence(self) -> "ProviderResult":
        if self.status in {"verified", "rejected"} and (
            not self.modelName or not self.modelVersion
        ):
            raise ValueError("Missing model provenance")
        if self.status == "verified" and not (
            self.livenessVerified
            and self.identityVerified
            and self.livenessScore is not None
            and self.faceMatchScore is not None
        ):
            raise ValueError("Identity and liveness decisions are required")
        if self.status != "verified" and self.livenessVerified and self.identityVerified:
            raise ValueError("Contradictory provider decision")
        return self
