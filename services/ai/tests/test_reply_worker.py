import unittest
from types import SimpleNamespace

from app.reply.extraction import ConversationSummarizer
from app.reply.service import ReplyAIService
from app.reply.worker import RESULTS_QUEUE, make_processor
from tests.reply_helpers import SETTINGS, FakeEmbedder, indexed
from tests.test_reply_api import chunk_payload


class FakeQueue:
    """假的 BullMQ Queue：只記錄 add 被呼叫時的參數。"""

    def __init__(self):
        self.added = []

    async def add(self, name, data, opts=None):
        self.added.append((name, data, opts))


class ReplyWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.results = FakeQueue()
        service = ReplyAIService(SETTINGS, embedder=FakeEmbedder(), summarizer=ConversationSummarizer(SETTINGS))
        self.process = make_processor(service, self.results)

    async def test_job_result_goes_to_results_queue(self):
        job = SimpleNamespace(name="chunk-embed", id="job-1", data=chunk_payload(True))
        returned = await self.process(job, None)
        self.assertEqual(returned, {"ok": True, "resultQueue": RESULTS_QUEUE})
        name, data, opts = self.results.added[0]
        self.assertEqual((name, data["sourceJobId"], data["key"]), ("chunk-embed", "job-1", "c-1"))
        self.assertEqual(len(data["result"]["chunks"]), 2)
        self.assertEqual(opts["attempts"], 5)

    async def test_bad_jobs_raise_fixed_codes_without_content(self):
        with self.assertRaisesRegex(ValueError, "^UNSUPPORTED_JOB$"):
            await self.process(SimpleNamespace(name="unknown", id="2", data={}), None)
        with self.assertRaisesRegex(ValueError, "^INVALID_JOB_INPUT$"):
            await self.process(SimpleNamespace(name="chunk-embed", id="3", data={"secret": "聊天內容"}), None)
        summary_job = SimpleNamespace(
            name="summarize", id="4", data={"conversationId": "c-1", "messages": [indexed("u-a", "嗨", 0).model_dump(mode="json")]}
        )
        with self.assertRaisesRegex(RuntimeError, "^EXTRACTION_NOT_CONFIGURED$"):
            await self.process(summary_job, None)
        self.assertEqual(self.results.added, [])


if __name__ == "__main__":
    unittest.main()
