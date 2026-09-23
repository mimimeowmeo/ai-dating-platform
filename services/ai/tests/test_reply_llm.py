import asyncio
import unittest

import httpx
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel

from app.config import Settings
from app.reply.errors import AIServiceError
from app.reply.llm import TimeoutModel, build_chain, describe_failure, run_agent
import tests.reply_helpers  # noqa: F401  匯入即關閉真實模型請求


async def slow(messages, info):
    """模擬很慢、但不會回錯誤的模型（例如當時 60 秒才回應的 gemini-3.5-flash-lite）。"""
    await asyncio.sleep(1)
    return ModelResponse(parts=[TextPart("slow")])


def fast(messages, info):
    """模擬正常回應的模型。"""
    return ModelResponse(parts=[TextPart("fast")])


def unreachable(messages, info):
    """模擬 DNS 暫時失敗：SDK 直接丟出 httpx 的網路錯誤，Pydantic AI 不會替 Gemini 轉換它。"""
    raise httpx.ConnectError("[Errno -5] No address associated with hostname")


class PerModelTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_model_times_out_and_next_model_takes_over(self):
        chain = FallbackModel(
            TimeoutModel(FunctionModel(slow, model_name="slow"), 0.1),
            TimeoutModel(FunctionModel(fast, model_name="fast"), 0.1),
        )
        result = await Agent().run("hi", model=chain)
        self.assertEqual((result.output, result.response.model_name), ("fast", "fast"))

    async def test_network_error_falls_back_to_next_model(self):
        # 以前網路錯誤會原樣往外丟：不換備援模型，服務還會回 500。
        chain = FallbackModel(
            TimeoutModel(FunctionModel(unreachable, model_name="gemini"), 5),
            TimeoutModel(FunctionModel(fast, model_name="fast"), 5),
        )
        result = await Agent().run("hi", model=chain)
        self.assertEqual((result.output, result.response.model_name), ("fast", "fast"))

    async def test_whole_chain_failure_is_logged_without_chat_content(self):
        chain = FallbackModel(
            TimeoutModel(FunctionModel(slow, model_name="ollama"), 0.05),
            TimeoutModel(FunctionModel(unreachable, model_name="gemini"), 5),
        )
        with self.assertLogs("app.reply.llm", level="WARNING") as logs:
            with self.assertRaises(AIServiceError) as caught:
                await run_agent(Agent(), "我的私密聊天內容", chain, 5, "LLM_NOT_CONFIGURED", "LLM_UNAVAILABLE")
        self.assertEqual(caught.exception.code, "LLM_UNAVAILABLE")
        self.assertEqual(
            logs.output,
            ["WARNING:app.reply.llm:LLM_UNAVAILABLE: ollama: timed out after 0.05s → gemini: network error: ConnectError"],
        )
        self.assertNotIn("私密", "\n".join(logs.output))

    def test_describe_failure_keeps_status_code_but_not_the_body(self):
        error = ModelHTTPError(503, "gemini-3.8-flash", {"error": {"message": "可能夾帶 prompt 的內容"}})
        self.assertEqual(describe_failure(error), "gemini-3.8-flash: HTTP 503")

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
