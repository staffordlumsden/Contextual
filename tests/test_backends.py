import io
import json
import unittest
from unittest.mock import patch

import contextual2


class BackendTests(unittest.TestCase):
    def test_litert_parameter_translation(self):
        model = contextual2.ChatModel("litert", "gemma4-26b")
        body = contextual2.litert_request_body(
            model,
            [{"role": "user", "content": "Hello"}],
            {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 7,
                "num_predict": 128,
                "repeat_penalty": 1.1,
            },
            thinking_enabled=False,
            stream=True,
        )

        self.assertEqual(body["model"], "gemma4-26b")
        self.assertEqual(body["max_completion_tokens"], 128)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("repeat_penalty", body)

    def test_litert_request_has_bounded_default_output(self):
        model = contextual2.ChatModel("litert", "gemma4-26b")

        body = contextual2.litert_request_body(
            model,
            [{"role": "user", "content": "Hello"}],
            {},
            thinking_enabled=False,
            stream=True,
        )

        self.assertEqual(
            body["max_completion_tokens"],
            contextual2.LITERT_DEFAULT_MAX_COMPLETION_TOKENS,
        )

    @patch("contextual2.urllib.request.urlopen")
    def test_litert_model_discovery(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({
            "data": [{"id": "gemma4-26b"}, {"id": "gemma4-12b"}],
        }).encode("utf-8"))

        self.assertEqual(contextual2.list_litert_models(), ["gemma4-26b", "gemma4-12b"])

    @patch("contextual2.urllib.request.urlopen")
    def test_litert_stream_is_normalised_for_the_ui(self, urlopen):
        events = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        ]
        stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        stream += "data: [DONE]\n\n"
        urlopen.return_value = io.BytesIO(stream.encode("utf-8"))

        model = contextual2.ChatModel("litert", "gemma4-26b")
        chunks = list(contextual2.litert_chat_stream(
            model,
            [{"role": "user", "content": "Hello"}],
            {},
            thinking_enabled=False,
        ))

        self.assertEqual("".join(chunk["message"]["content"] for chunk in chunks), "Hello world")
        self.assertTrue(chunks[-1]["done"])
        self.assertEqual(chunks[-1]["prompt_eval_count"], 5)
        self.assertEqual(chunks[-1]["eval_count"], 2)

    @patch("contextual2.urllib.request.urlopen")
    def test_litert_rejects_reserved_gemma_tokens(self, urlopen):
        event = {"choices": [{"delta": {"content": "[multimodal]<unused30>"}}]}
        stream = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        urlopen.return_value = io.BytesIO(stream.encode("utf-8"))
        model = contextual2.ChatModel("litert", "gemma4-26b")

        with self.assertRaisesRegex(RuntimeError, "reserved Gemma control tokens"):
            list(contextual2.litert_chat_stream(
                model,
                [{"role": "user", "content": "Hello"}],
                {},
                thinking_enabled=False,
            ))

    @patch("contextual2.urllib.request.urlopen")
    def test_litert_reports_exhausted_thinking_budget(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({
            "choices": [{"message": {"role": "assistant", "content": None}}],
            "usage": {
                "completion_tokens": 64,
                "completion_tokens_details": {"reasoning_tokens": 61},
            },
        }).encode("utf-8"))
        model = contextual2.ChatModel("litert", "gemma4-26b")

        with self.assertRaisesRegex(RuntimeError, "Increase num_predict"):
            contextual2.chat_once(
                model,
                [{"role": "user", "content": "Think carefully"}],
                {"num_predict": 64},
                thinking_enabled=True,
            )

    def test_backend_parameter_support(self):
        litert = contextual2.ChatModel("litert", "gemma4-26b")
        ollama = contextual2.ChatModel("ollama", "qwen3:8b")

        self.assertNotIn("repeat_penalty", contextual2.supported_parameters(litert))
        self.assertIn("repeat_penalty", contextual2.supported_parameters(ollama))
        self.assertIn("temperature", contextual2.supported_parameters(litert))

    @patch("contextual2.Prompt.ask", return_value="2")
    @patch("contextual2.list_litert_models", return_value=["gemma4-26b"])
    @patch("contextual2.ollama.list")
    def test_grouped_picker_selection_carries_backend(self, ollama_list, _litert_models, _prompt):
        ollama_list.return_value = {"models": [{"model": "qwen3:4b-instruct"}]}

        selected = contextual2.select_chat_model(is_interactive=True)

        self.assertEqual(selected, contextual2.ChatModel("litert", "gemma4-26b"))

    @patch.object(contextual2, "LITERT_CONTEXT_WINDOW_TOKENS", 1024)
    def test_litert_context_budget_preserves_system_and_latest_user_message(self):
        messages = [
            {"role": "system", "content": "Keep citations."},
            {"role": "user", "content": "A" * 1500},
            {"role": "assistant", "content": "B" * 1500},
            {"role": "user", "content": "Latest question"},
        ]

        prepared = contextual2.prepare_litert_messages(messages, {"num_predict": 512})

        self.assertEqual(prepared[0], messages[0])
        self.assertEqual(prepared[-1], messages[-1])
        self.assertNotEqual(prepared[1].get("content"), "B" * 1500)


if __name__ == "__main__":
    unittest.main()
