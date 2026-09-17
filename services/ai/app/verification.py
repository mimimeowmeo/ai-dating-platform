import asyncio
import base64
import binascii
import io
import json
import warnings

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .config import Settings
from .schemas import MAX_IMAGE_BYTES, ProviderResult, VerificationRequest, VerificationResult

FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MIN_DIMENSION = 64
MAX_DIMENSION = 4096
MAX_PIXELS = 16_000_000
MAX_PROVIDER_BYTES = 16 * 1024


class InvalidImage(ValueError):
    pass


def validate_image(request: VerificationRequest) -> None:
    """驗證完整解碼內容，不把影像落地、不輸出 EXIF 或生物特徵。"""
    try:
        data = base64.b64decode(request.imageBase64, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidImage("INVALID_IMAGE") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage("IMAGE_TOO_LARGE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if FORMATS.get(image.format) != request.mimeType:
                    raise InvalidImage("IMAGE_TYPE_MISMATCH")
                if (
                    min(width, height) < MIN_DIMENSION
                    or max(width, height) > MAX_DIMENSION
                    or width * height > MAX_PIXELS
                ):
                    raise InvalidImage("INVALID_IMAGE_DIMENSIONS")
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("ANIMATED_IMAGE_NOT_ALLOWED")
                image.verify()
            # verify 檢查結構，load 確認壓縮像素可實際完整解碼。
            with Image.open(io.BytesIO(data)) as image:
                image.load()
    except InvalidImage:
        raise
    except (
        UnidentifiedImageError, OSError, SyntaxError, ValueError,
        Image.DecompressionBombError, Image.DecompressionBombWarning,
    ):
        raise InvalidImage("INVALID_IMAGE") from None


def unavailable(reason: str) -> VerificationResult:
    return VerificationResult(status="unavailable", reasonCode=reason)


class VerificationService:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        await asyncio.to_thread(validate_image, request)
        if not self.settings.provider_url:
            return unavailable("MODEL_NOT_CONFIGURED")
        if not self.settings.provider_configured:
            return unavailable("PROVIDER_CONFIGURATION_INVALID")
        try:
            # 同時限制整個請求時間，避免 provider 持續零碎回應延長等待。
            async with asyncio.timeout(self.settings.provider_timeout_seconds):
                async with httpx.AsyncClient(
                    transport=self.transport,
                    timeout=self.settings.provider_timeout_seconds,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    async with client.stream(
                        "POST", self.settings.provider_url,
                        json=request.model_dump(),
                        headers={
                            "Authorization": f"Bearer {self.settings.provider_token}",
                            "Accept": "application/json",
                        },
                    ) as response:
                        if response.status_code != 200:
                            return unavailable("PROVIDER_UNAVAILABLE")
                        if response.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                            return unavailable("PROVIDER_INVALID_RESPONSE")
                        payload = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(payload) + len(chunk) > MAX_PROVIDER_BYTES:
                                return unavailable("PROVIDER_INVALID_RESPONSE")
                            payload.extend(chunk)
            result = ProviderResult.model_validate_json(payload)
            return VerificationResult.model_validate(result.model_dump(exclude={
                "verificationType", "livenessVerified", "identityVerified",
            }))
        except (httpx.TimeoutException, TimeoutError):
            return unavailable("PROVIDER_TIMEOUT")
        except (httpx.HTTPError, OSError):
            return unavailable("PROVIDER_UNAVAILABLE")
        except (ValidationError, json.JSONDecodeError, UnicodeError, ValueError):
            return unavailable("PROVIDER_INVALID_RESPONSE")
