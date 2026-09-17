import asyncio
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.schemas import MAX_BODY_BYTES, VerificationRequest
from app.verification import VerificationService
from app.worker import make_processor, maintain_heartbeat


def image_request(size=(128, 128), format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (50, 80, 100)).save(buffer, format=format)
    return {
        "imageBase64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mimeType": {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[format],
        "requestId": "test-request-1",
    }


def decision(**overrides):
    return {
        "status": "verified", "reasonCode": "VERIFICATION_PASSED",
        "modelName": "test-provider", "modelVersion": "test-version",
        "verificationType": "identity_verification",
        "identityVerified": True, "livenessVerified": True,
        "livenessScore": 0.9, "faceMatchScore": 0.9,
        **overrides,
    }


SETTINGS = Settings(internal_token="internal-test-secret")
PROVIDER_SETTINGS = Settings(
    internal_token="internal-test-secret", provider_url="https://provider.invalid/verify",
    provider_token="provider-test-secret",
)
HEADERS = {"X-Internal-Token": SETTINGS.internal_token}
ENDPOINT = "/internal/ai/face/verify"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(SETTINGS))

    def test_health_without_auth(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verificationProvider"], "unavailable")

    def test_auth_checked_before_bad_body(self):
        response = self.client.post(ENDPOINT, content="not json")
        self.assertEqual(response.status_code, 401)
        response = self.client.post(ENDPOINT, json=image_request(), headers={"X-Internal-Token": "wrong"})
        self.assertEqual(response.status_code, 401)

    def test_missing_configured_token_disables_internal_api(self):
        client = TestClient(create_app(Settings()))
        self.assertEqual(client.post(ENDPOINT, json=image_request()).status_code, 503)

    def test_valid_image_never_passes_without_model(self):
        for format in ("PNG", "JPEG", "WEBP"):
            with self.subTest(format=format):
                response = self.client.post(ENDPOINT, json=image_request(format=format), headers=HEADERS)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "unavailable")
                self.assertEqual(response.json()["reasonCode"], "MODEL_NOT_CONFIGURED")
                self.assertIsNone(response.json()["modelName"])

    def test_bad_base64_and_html_cannot_pass(self):
        for content in ("????", base64.b64encode(b"<html>not a photo</html>").decode()):
            with self.subTest(content=content):
                request = {**image_request(), "imageBase64": content}
                response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["code"], "INVALID_IMAGE")
                self.assertNotIn(content, response.text)

    def test_mime_mismatch(self):
        request = {**image_request(), "mimeType": "image/jpeg"}
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "IMAGE_TYPE_MISMATCH")

    def test_unsafe_dimensions(self):
        for size in ((10, 10), (4097, 64)):
            response = self.client.post(ENDPOINT, json=image_request(size=size), headers=HEADERS)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], "INVALID_IMAGE_DIMENSIONS")

    def test_truncated_pixels_rejected(self):
        request = image_request(format="JPEG")
        raw = base64.b64decode(request["imageBase64"])
        request["imageBase64"] = base64.b64encode(raw[:-20]).decode()
        self.assertEqual(self.client.post(ENDPOINT, json=request, headers=HEADERS).status_code, 400)

    def test_animated_image_rejected(self):
        buffer = io.BytesIO()
        Image.new("RGB", (128, 128), "red").save(
            buffer, format="PNG", save_all=True, append_images=[Image.new("RGB", (128, 128), "blue")],
        )
        request = {**image_request(), "imageBase64": base64.b64encode(buffer.getvalue()).decode()}
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "ANIMATED_IMAGE_NOT_ALLOWED")

    def test_validation_does_not_echo_sensitive_input(self):
        request = {**image_request(), "untrustedField": "secret-selfie-content"}
        response = self.client.post(ENDPOINT, json=request, headers=HEADERS)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(request["imageBase64"], response.text)
        self.assertNotIn("secret-selfie-content", response.text)

    def test_http_body_limit_even_without_content_length(self):
        def chunks():
            for _ in range(8):
                yield b"a" * (1024 * 1024)
        response = self.client.post(ENDPOINT, content=chunks(), headers=HEADERS)
        self.assertGreater(8 * 1024 * 1024, MAX_BODY_BYTES)
        self.assertEqual(response.status_code, 413)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def get_result(self, payload, status_code=200):
        service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(lambda _: httpx.Response(status_code, json=payload)))
        return await service.verify(VerificationRequest(**image_request()))

    async def test_verified_requires_identity_and_liveness_evidence(self):
        cases = [
            {"status": "verified", "faceDetected": True},
            decision(verificationType="face_detection"),
            decision(identityVerified=False), decision(livenessVerified=False),
            decision(livenessVerified="true"), decision(livenessScore=None),
            decision(modelVersion=None), decision(faceMatchScore=1.1),
            decision(faceMatchScore="0.99"), decision(livenessScore=float("inf")),
            decision(status="pending"), decision(embedding=[1, 2, 3]),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                # 手動 JSON 允許非有限值，模擬不符合 JSON／schema 的上游。
                service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(
                    lambda _, p=payload: httpx.Response(200, content=json.dumps(p), headers={"Content-Type": "application/json"})
                ))
                result = await service.verify(VerificationRequest(**image_request()))
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.reasonCode, "PROVIDER_INVALID_RESPONSE")

    async def test_complete_provider_decision_forwarded_without_extra_fields(self):
        result = await self.get_result(decision())
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.modelVersion, "test-version")
        self.assertEqual(result.faceMatchScore, 0.9)
        self.assertNotIn("identityVerified", result.model_dump())

    async def test_provider_rejection_preserved_without_new_threshold(self):
        result = await self.get_result(decision(
            status="rejected", reasonCode="LIVENESS_FAILED", livenessVerified=False,
            livenessScore=0.99,
        ))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.reasonCode, "LIVENESS_FAILED")

    async def test_provider_request_contract(self):
        def respond(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["Authorization"], "Bearer provider-test-secret")
            self.assertEqual(json.loads(request.content), image_request())
            return httpx.Response(200, json=decision())
        result = await VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(respond)).verify(VerificationRequest(**image_request()))
        self.assertEqual(result.status, "verified")

    async def test_disabled_provider_has_no_network_call(self):
        def unexpected(_request):
            self.fail("Unconfigured model must not make a network call")
        result = await VerificationService(SETTINGS, httpx.MockTransport(unexpected)).verify(VerificationRequest(**image_request()))
        self.assertEqual(result.reasonCode, "MODEL_NOT_CONFIGURED")

    async def test_partial_provider_config_fails_closed(self):
        result = await VerificationService(Settings(provider_url="https://example.com")).verify(VerificationRequest(**image_request()))
        self.assertEqual(result.reasonCode, "PROVIDER_CONFIGURATION_INVALID")

    async def test_provider_timeout(self):
        def timeout(_request):
            raise httpx.ReadTimeout("sensitive-provider-error-do-not-return")
        result = await VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(timeout)).verify(VerificationRequest(**image_request()))
        self.assertEqual(result.reasonCode, "PROVIDER_TIMEOUT")
        self.assertNotIn("sensitive", result.model_dump_json())

    async def test_provider_redirect_and_error_fail_closed(self):
        for status in (302, 401, 500):
            result = await self.get_result(decision(), status_code=status)
            self.assertEqual(result.status, "unavailable")

    async def test_non_json_and_oversized_provider_body(self):
        for response in (httpx.Response(200, text="OK"), httpx.Response(200, json={"huge": "x" * 20000})):
            service = VerificationService(PROVIDER_SETTINGS, httpx.MockTransport(lambda _: response))
            result = await service.verify(VerificationRequest(**image_request()))
            self.assertEqual(result.reasonCode, "PROVIDER_INVALID_RESPONSE")


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_consumes_same_contract_and_redacts(self):
        job = SimpleNamespace(name="verify", data=image_request(), updateData=AsyncMock())
        result = await make_processor(VerificationService(SETTINGS))(job, "lock-token")
        job.updateData.assert_awaited_once_with({"redacted": True})
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("imageBase64", result)
        self.assertNotIn("faceMatchScore", result)

    async def test_unknown_jobs_rejected(self):
        job = SimpleNamespace(name="embed-conversation", data={}, updateData=AsyncMock())
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_JOB"):
            await make_processor(VerificationService(SETTINGS))(job, "lock-token")

    async def test_failed_jobs_never_include_input_in_error(self):
        job = SimpleNamespace(name="verify", data={"private": "sensitive-image"}, updateData=AsyncMock())
        with self.assertRaises(ValueError) as caught:
            await make_processor(VerificationService(SETTINGS))(job, "lock-token")
        self.assertEqual(str(caught.exception), "INVALID_VERIFICATION_INPUT")

    async def test_heartbeat_requires_redis_and_worker(self):
        for redis_available, running in ((True, True), (False, True), (True, False)):
            with self.subTest(redis_available=redis_available, running=running), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "heartbeat"
                stopped = asyncio.Event()

                async def ping():
                    stopped.set()
                    if not redis_available:
                        raise OSError("not connected")

                client = SimpleNamespace(ping=ping)
                worker = SimpleNamespace(running=running, closing=False)
                await maintain_heartbeat(client, worker, stopped, path)
                self.assertEqual(path.exists(), redis_available and running)


if __name__ == "__main__":
    unittest.main()
