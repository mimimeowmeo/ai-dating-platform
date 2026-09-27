"""worker 的 job 處理：先把照片從 Redis 清掉、驗證輸入、結果放進 rec-results、錯誤只丟固定代碼；
心跳：正在跑時寫在線訊號（NestJS 看到才排照片），關閉中就撤掉。"""
import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from appearance.pipeline import MODEL_VERSION, InvalidImage
from app.worker import JOB_NAME, ONLINE_KEY, RESULT_JOB_OPTIONS, maintain_heartbeat, make_processor

PHOTO_ID = "5f0c2b1e-8a3d-4c7e-9b1a-2d3e4f5a6b7c"
IMAGE = base64.b64encode(b"jpeg-bytes").decode()


class FakeJob:
    def __init__(self, data, name=JOB_NAME):
        self.name = name
        self.data = data
        self.updates = []

    async def updateData(self, data):
        self.updates.append(data)
        self.data = data


class FakeResults:
    def __init__(self):
        self.added = []

    async def add(self, name, data, options):
        self.added.append((name, data, options))


class FakeEmbedder:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.images = []

    def embed(self, image):
        self.images.append(image)
        if self.error:
            raise self.error
        return self.result


class ProcessTest(unittest.IsolatedAsyncioTestCase):
    async def run_job(self, job, embedder=None):
        results = FakeResults()
        process = make_processor(embedder or FakeEmbedder([0.1] * 512), results)
        return await process(job, "token"), results

    async def test_embeds_and_posts_result(self):
        embedder = FakeEmbedder([0.25] * 512)
        job = FakeJob({"photoId": PHOTO_ID, "image": IMAGE})
        value, results = await self.run_job(job, embedder)
        self.assertEqual(job.updates, [{"redacted": True}])
        self.assertEqual(embedder.images, [b"jpeg-bytes"])
        self.assertEqual(
            results.added,
            [(JOB_NAME, {"photoId": PHOTO_ID, "modelVersion": MODEL_VERSION, "embedding": [0.25] * 512}, RESULT_JOB_OPTIONS)],
        )
        self.assertEqual(value, {"ok": True, "faceFound": True})

    async def test_no_face_posts_null_embedding(self):
        value, results = await self.run_job(FakeJob({"photoId": PHOTO_ID, "image": IMAGE}), FakeEmbedder(None))
        self.assertIsNone(results.added[0][1]["embedding"])
        self.assertEqual(value["faceFound"], False)

    async def test_unsupported_job_is_redacted_and_rejected(self):
        job = FakeJob({"photoId": PHOTO_ID, "image": IMAGE}, name="chunk-embed")
        with self.assertRaisesRegex(ValueError, "^UNSUPPORTED_JOB$"):
            await self.run_job(job)
        self.assertEqual(job.data, {"redacted": True})

    async def test_invalid_input_raises_fixed_code(self):
        for data in ({"photoId": "not-a-uuid", "image": IMAGE}, {"photoId": PHOTO_ID}, {"photoId": PHOTO_ID, "image": IMAGE, "x": 1}):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "^INVALID_JOB_INPUT$"):
                await self.run_job(FakeJob(data))

    async def test_bad_image_raises_fixed_code(self):
        with self.assertRaisesRegex(ValueError, "^INVALID_IMAGE$"):
            await self.run_job(FakeJob({"photoId": PHOTO_ID, "image": "%%% not base64"}))
        with self.assertRaisesRegex(ValueError, "^INVALID_IMAGE$"):
            await self.run_job(FakeJob({"photoId": PHOTO_ID, "image": IMAGE}), FakeEmbedder(error=InvalidImage("x")))


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.expiry = {}

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.expiry[key] = ex

    async def delete(self, key):
        self.values.pop(key, None)


class FakeWorker:
    def __init__(self, running=True, closing=False):
        self.running = running
        self.closing = closing


class HeartbeatTest(unittest.IsolatedAsyncioTestCase):
    async def beat(self, client, worker):
        stopped = asyncio.Event()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heartbeat"
            task = asyncio.create_task(maintain_heartbeat(client, worker, stopped, path))
            await asyncio.sleep(0.05)
            written = path.exists()
            stopped.set()
            await task
            return written, path.exists()

    async def test_running_worker_announces_itself(self):
        client = FakeRedis()
        written, left = await self.beat(client, FakeWorker())
        self.assertEqual(client.values, {ONLINE_KEY: "1"})
        self.assertEqual(client.expiry[ONLINE_KEY], 60)
        self.assertTrue(written)
        self.assertFalse(left)

    async def test_closing_worker_withdraws(self):
        client = FakeRedis({ONLINE_KEY: "1"})
        written, _ = await self.beat(client, FakeWorker(closing=True))
        self.assertEqual(client.values, {})
        self.assertFalse(written)


if __name__ == "__main__":
    unittest.main()
