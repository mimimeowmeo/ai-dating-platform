import unittest

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.reply.extraction import ConversationSummarizer, StyleProfileBuilder
from app.reply.service import ReplyAIService
from app.reply.suggest import ReplySuggester
from tests.reply_helpers import HEADERS, SETTINGS, FakeEmbedder, indexed, json_model
from tests.test_reply_extraction import MAP_OUTPUT
from tests.test_reply_suggest import DRAFTS, make_request


def fake_service() -> ReplyAIService:
    embedder = FakeEmbedder()
    return ReplyAIService(
        SETTINGS,
        embedder=embedder,
        suggester=ReplySuggester(SETTINGS, model=json_model({"suggestions": DRAFTS})),
        style_builder=StyleProfileBuilder(SETTINGS, model=json_model(MAP_OUTPUT), embedder=embedder),
        summarizer=ConversationSummarizer(SETTINGS, model=json_model({"summary": "兩人聊到爬山"})),
    )


def chunk_payload(embed: bool) -> dict:
    messages = [indexed("u-a", "嗨", 0), indexed("u-b", "哈囉", 1), indexed("u-a", "早安", 60)]
    return {"conversationId": "c-1", "messages": [m.model_dump(mode="json") for m in messages], "embed": embed}


class ReplyApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(SETTINGS, reply_service=fake_service()))

    def test_internal_token_required(self):
        response = self.client.post("/internal/ai/embed", json={"texts": ["嗨"]})
        self.assertEqual(response.status_code, 401)

    def test_embed(self):
        response = self.client.post("/internal/ai/embed", json={"texts": ["爬山", "貓"], "purpose": "query"}, headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["model"], body["dimensions"], len(body["vectors"])), ("fake-embedding", 4, 2))

    def test_chunks_with_and_without_vectors(self):
        body = self.client.post("/internal/ai/chunks", json=chunk_payload(True), headers=HEADERS).json()
        self.assertEqual([chunk["messageCount"] for chunk in body["chunks"]], [2, 1])
        self.assertTrue(all(chunk["vector"] for chunk in body["chunks"]))
        self.assertEqual((body["chunkVersion"], body["embeddingModel"]), ("chunk-v1", "fake-embedding"))
        body = self.client.post("/internal/ai/chunks", json=chunk_payload(False), headers=HEADERS).json()
        self.assertTrue(all(chunk["vector"] is None for chunk in body["chunks"]))

    def test_topic_spans(self):
        messages = [
            indexed("u-a", "你喜歡爬山嗎？", 0, origin="ai_verbatim", intent="question"),
            indexed("u-b", "喜歡山", 1), indexed("u-a", "哪座山", 2), indexed("u-b", "合歡山", 3),
            indexed("u-b", "我養貓", 4), indexed("u-a", "貓好可愛", 5), indexed("u-b", "貓超黏人", 6),
            indexed("u-b", "你養貓嗎", 7), indexed("u-a", "沒有貓", 8), indexed("u-b", "貓咪", 9),
        ]
        payload = {"conversationId": "c-1", "messages": [m.model_dump(mode="json") for m in messages]}
        body = self.client.post("/internal/ai/topic-spans", json=payload, headers=HEADERS).json()
        self.assertTrue(body["usedVectors"])
        self.assertEqual(len(body["spans"]), 1)
        span = body["spans"][0]
        self.assertEqual((span["initiatorId"], span["endReason"], span["endMessageId"]), ("u-a", "topic_shift", messages[3].id))

    def test_reply_suggestions_summary_and_style_profile(self):
        response = self.client.post(
            "/internal/ai/reply-suggestions", json=make_request().model_dump(mode="json"), headers=HEADERS
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["suggestions"]), 3)

        summary = {"conversationId": "c-1", "messages": [indexed("u-a", "嗨", 0).model_dump(mode="json")]}
        response = self.client.post("/internal/ai/conversation-summary", json=summary, headers=HEADERS)
        self.assertEqual(response.json()["summary"], "兩人聊到爬山")

        profile = {"userId": "u-b", "bio": "喜歡爬山跟拍照，週末常往山上跑喔"}
        response = self.client.post("/internal/ai/style-profile", json=profile, headers=HEADERS)
        self.assertEqual(response.json()["card"]["sampleSource"], "bio")

    def test_invalid_request_does_not_echo_input(self):
        response = self.client.post(
            "/internal/ai/embed", json={"texts": ["秘密內容"], "unknown": True}, headers=HEADERS
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INVALID_REQUEST")
        self.assertNotIn("秘密內容", response.text)

    def test_unconfigured_models_return_503_codes(self):
        client = TestClient(create_app(Settings(internal_token=SETTINGS.internal_token)))
        response = client.post(
            "/internal/ai/reply-suggestions", json=make_request().model_dump(mode="json"), headers=HEADERS
        )
        self.assertEqual((response.status_code, response.json()["code"]), (503, "LLM_NOT_CONFIGURED"))
        response = client.post("/internal/ai/embed", json={"texts": ["嗨"]}, headers=HEADERS)
        self.assertEqual((response.status_code, response.json()["code"]), (503, "EMBEDDING_NOT_CONFIGURED"))
        health = client.get("/health").json()
        self.assertEqual(health["replySuggestions"]["replyModels"], "unavailable")

    def test_health_reports_configured_components(self):
        health = self.client.get("/health").json()
        self.assertEqual(health["replySuggestions"], {"replyModels": "configured", "embedding": "configured", "extraction": "configured"})


class ConfigTests(unittest.TestCase):
    def test_model_lists_and_ollama_url_from_env(self):
        from unittest.mock import patch

        env = {
            "GEMINI_API_KEY": "k",
            "AI_REPLY_MODELS": "gemini-3.8-flash, ollama:gemma4:12b",
            "AI_EXTRACTION_MODELS": "-",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "AI_EMBEDDING_DIMENSIONS": "abc",
        }
        with patch.dict("os.environ", env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.reply_models, ("gemini-3.8-flash", "ollama:gemma4:12b"))
        self.assertEqual(settings.extraction_models, ())
        self.assertEqual(settings.embedding_dimensions, 768)  # 格式錯誤 → 預設值
        self.assertTrue(settings.gemini_configured and settings.ollama_configured)

    def test_build_chain_skips_unconfigured_models(self):
        from pydantic_ai.models.fallback import FallbackModel

        from app.reply.llm import build_chain, normalize_ollama_base_url

        self.assertEqual(normalize_ollama_base_url("http://localhost:11434/"), "http://localhost:11434/v1")
        self.assertIsNone(build_chain(("gemini-3.8-flash", "ollama:gemma4:12b"), SETTINGS, "reply"))
        both = Settings(gemini_api_key="k", ollama_base_url="http://localhost:11434")
        self.assertIsInstance(build_chain(("gemini-3.8-flash", "ollama:gemma4:12b"), both, "reply"), FallbackModel)
        only_ollama = Settings(ollama_base_url="http://localhost:11434")
        model = build_chain(("gemini-3.8-flash", "ollama:gemma4:12b"), only_ollama, "extraction")
        self.assertEqual(model.model_name, "gemma4:12b")

    def test_ollama_cloud_needs_key_and_uses_tool_output(self):
        from unittest.mock import patch

        from pydantic_ai import NativeOutput, ToolOutput

        from app.reply.llm import build_chain, structured_output
        from app.reply.schemas import DraftBatch

        specs = ("ollama:gemma4:31b",)
        local = Settings(ollama_base_url="http://localhost:11434/v1")
        cloud_without_key = Settings(ollama_base_url="https://ollama.com/v1")
        cloud = Settings(ollama_base_url="https://ollama.com/v1", ollama_api_key="k")
        self.assertEqual((local.ollama_is_cloud, cloud.ollama_is_cloud), (False, True))
        self.assertIsNone(build_chain(specs, cloud_without_key, "extraction"))  # 雲端沒有 key → 視為未設定
        self.assertEqual(build_chain(specs, cloud, "extraction").model_name, "gemma4:31b")
        self.assertIsInstance(structured_output(DraftBatch, specs, local), NativeOutput)
        self.assertIsInstance(structured_output(DraftBatch, specs, cloud), ToolOutput)
        self.assertIsInstance(structured_output(DraftBatch, ("gemini-3.8-flash",), cloud), NativeOutput)
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "https://ollama.com/v1", "OLLAMA_API_KEY": "k"}, clear=True):
            self.assertTrue(Settings.from_env().ollama_configured)


if __name__ == "__main__":
    unittest.main()
