import asyncio
import unittest

from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel

from app.config import Settings
from app.reply.errors import AIServiceError
from app.reply.llm import TimeoutModel, build_chain, run_agent
import tests.reply_helpers  # noqa: F401  匯入即關閉真實模型請求


async def slow(messages, info):
    """模擬很慢、但不會回錯誤的模型（例如當時 60 秒才回應的 gemini-3.5-flash-lite）。"""
    await asyncio.sleep(1)
    return ModelResponse(parts=[TextPart("slow")])


def fast(messages, info):
    """模擬正常回應的模型。"""
    return ModelResponse(parts=[TextPart("fast")])


class PerModelTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_model_times_out_and_next_model_takes_over(self):
        chain = FallbackModel(
            TimeoutModel(FunctionModel(slow, model_name="slow"), 0.1),
            TimeoutModel(FunctionModel(fast, model_name="fast"), 0.1),
        )
        result = await Agent().run("hi", model=chain)
        self.assertEqual((result.output, result.response.model_name), ("fast", "fast"))

    async def test_single_slow_model_becomes_unavailable_before_total_timeout(self):
        model = TimeoutModel(FunctionModel(slow, model_name="slow"), 0.1)
        with self.assertRaises(AIServiceError) as caught:
            await run_agent(Agent(), "hi", model, 5, "LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE")
        self.assertEqual(caught.exception.code, "LLM_UNAVAILABLE")

    def test_build_chain_wraps_every_model_with_its_purpose_timeout(self):
        settings = Settings(
            gemini_api_key="k",
            ollama_base_url="http://localhost:11434",
            reply_model_timeout_seconds=7,
            extraction_timeout_seconds=90,
        )
        chain = build_chain(("ollama:gemma4:31b", "gemini-3.8-flash"), settings, "reply")
        self.assertIsInstance(chain, FallbackModel)
        self.assertEqual([(type(m), m.timeout_seconds) for m in chain.models], [(TimeoutModel, 7), (TimeoutModel, 7)])
        single = build_chain(("ollama:gemma4:31b",), settings, "extraction")
        self.assertEqual((type(single), single.timeout_seconds, single.model_name), (TimeoutModel, 90, "gemma4:31b"))

    def test_default_reply_chain_prefers_ollama_then_gemini(self):
        self.assertEqual(Settings().reply_models, ("ollama:gemma4:31b", "gemini-3.8-flash"))


if __name__ == "__main__":
    unittest.main()
