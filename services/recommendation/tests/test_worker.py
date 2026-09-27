"""worker 的 job 處理：先把照片從 Redis 清掉、驗證輸入、結果放進 rec-results、錯誤只丟固定代碼；
心跳：正在跑時寫在線訊號（NestJS 看到才排照片），關閉中就撤掉。"""
import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from appearance.pipeline import MODEL_VERSION, InvalidImage
from app.worker import EXPLAIN_JOB, JOB_NAME, ONLINE_KEY, RESULT_JOB_OPTIONS, maintain_heartbeat, make_processor

PHOTO_ID = "5f0c2b1e-8a3d-4c7e-9b1a-2d3e4f5a6b7c"
ANCHOR_ID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
IMAGE = base64.b64encode(b"jpeg-bytes").decode()
ANCHOR_IMAGE = base64.b64encode(b"anchor-bytes").decode()
DIRECTION = [0] * 511 + [1]


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

    def explain(self, candidate, anchor, direction):
        self.images.append((candidate, anchor, direction))
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


class ExplainTest(unittest.IsolatedAsyncioTestCase):
    def job(self, **extra):
        data = {"requestId": f"{PHOTO_ID}:{ANCHOR_ID}", "candidate": IMAGE, "anchor": ANCHOR_IMAGE, **extra}
        return FakeJob(data, name=EXPLAIN_JOB)

    async def run_job(self, job, embedder):
        results = FakeResults()
        return await make_processor(embedder, results)(job, "token"), results

    async def test_explains_pair_and_posts_result(self):
        embedder = FakeEmbedder({"similarity": 0.9, "regions": []})
        job = self.job(direction=DIRECTION, directionVersion=MODEL_VERSION)
        value, results = await self.run_job(job, embedder)
        self.assertEqual(job.updates, [{"redacted": True}])
        self.assertEqual(embedder.images, [(b"jpeg-bytes", b"anchor-bytes", [0.0] * 511 + [1.0])])
        self.assertEqual(
            results.added,
            [(EXPLAIN_JOB, {"requestId": f"{PHOTO_ID}:{ANCHOR_ID}", "result": {"similarity": 0.9, "regions": []}}, RESULT_JOB_OPTIONS)],
        )
        self.assertTrue(value["explained"])

    async def test_direction_from_another_model_is_ignored(self):
        embedder = FakeEmbedder(None)
        value, results = await self.run_job(self.job(direction=DIRECTION, directionVersion="old"), embedder)
        self.assertIsNone(embedder.images[0][2])
        self.assertIsNone(results.added[0][1]["result"])
        self.assertFalse(value["explained"])

    async def test_invalid_input_raises_fixed_code(self):
        for extra in ({"requestId": PHOTO_ID}, {"direction": [0.1] * 3}, {"photoId": PHOTO_ID}):
            job = self.job(**extra)
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "^INVALID_JOB_INPUT$"):
                await self.run_job(job, FakeEmbedder())
            self.assertEqual(job.data, {"redacted": True})

    async def test_bad_image_raises_fixed_code(self):
        with self.assertRaisesRegex(ValueError, "^INVALID_IMAGE$"):
            await self.run_job(self.job(anchor="%%% not base64"), FakeEmbedder())
        with self.assertRaisesRegex(ValueError, "^INVALID_IMAGE$"):
            await self.run_job(self.job(), FakeEmbedder(error=InvalidImage("x")))


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
