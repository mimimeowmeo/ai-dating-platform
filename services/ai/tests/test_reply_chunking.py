import unittest

from app.reply.chunking import MAX_MESSAGES, build_chunks, format_line
from tests.reply_helpers import at, indexed


class ChunkingTests(unittest.TestCase):
    def test_time_gap_starts_new_chunk_without_overlap(self):
        messages = [indexed("u-a", "嗨", 0), indexed("u-b", "你好", 1), indexed("u-a", "早安", 40)]
        chunks = build_chunks(messages, now=at(500))
        self.assertEqual([len(chunk.messages) for chunk in chunks], [2, 1])
        self.assertEqual(chunks[1].messages[0].content, "早安")

    def test_message_cap_splits_with_two_message_overlap(self):
        messages = [indexed("u-a" if i % 2 else "u-b", f"第{i}則", i) for i in range(15)]
        chunks = build_chunks(messages, now=at(500))
        self.assertEqual(len(chunks[0].messages), MAX_MESSAGES)
        self.assertEqual([m.content for m in chunks[1].messages[:2]], ["第10則", "第11則"])
        self.assertEqual(chunks[1].messages[-1].content, "第14則")

    def test_token_cap_splits_long_messages(self):
        long_text = "好" * 150  # 約 195 tokens
        messages = [indexed("u-a", long_text, i) for i in range(3)]
        chunks = build_chunks(messages, now=at(500))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_estimate <= 450 for chunk in chunks))

    def test_last_chunk_open_only_when_recent(self):
        messages = [indexed("u-a", "嗨", 0), indexed("u-b", "哈囉", 1)]
        self.assertTrue(build_chunks(messages, now=at(10))[-1].is_open)
        self.assertFalse(build_chunks(messages, now=at(100))[-1].is_open)

    def test_format_line_uses_taipei_time_and_name(self):
        line = format_line(indexed("u-b", "你好\n呀", 0))
        self.assertEqual(line, "[09-01 20:00] 小美：你好 呀")

    def test_messages_are_sorted_before_chunking(self):
        messages = [indexed("u-b", "後面", 5), indexed("u-a", "前面", 1)]
        chunk = build_chunks(messages, now=at(500))[0]
        self.assertEqual([m.content for m in chunk.messages], ["前面", "後面"])


if __name__ == "__main__":
    unittest.main()
