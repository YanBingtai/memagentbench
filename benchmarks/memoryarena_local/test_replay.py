import unittest

from .replay import _chat_request_payload


class ChatRequestPayloadTest(unittest.TestCase):
    def test_qwen_uses_local_thinking_contract(self):
        payload = _chat_request_payload(
            "qwen3.8", [{"role": "user", "content": "hello"}], 0.2, 128, True
        )

        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("thinking", payload)

    def test_mimo_uses_official_thinking_contract(self):
        payload = _chat_request_payload(
            "mimo-v2.6-flash", [{"role": "user", "content": "hello"}], 0.2, 128, True
        )

        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    def test_mimo_can_disable_thinking_and_keep_temperature(self):
        payload = _chat_request_payload(
            "mimo-v2.6-flash", [{"role": "user", "content": "hello"}], 0.2, 128, False
        )

        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
