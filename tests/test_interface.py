import io
import unittest
from unittest.mock import patch

from rich.console import Console
from rich.segment import Segment
from rich.text import Text

import contextual2


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        resolver = patch("contextual2.context_window_size", return_value=4096)
        resolver.start()
        self.addCleanup(resolver.stop)

    def tearDown(self):
        contextual2.set_ui_theme("/dark")

    def test_theme_commands_are_exact_and_case_insensitive(self):
        self.assertTrue(contextual2.set_ui_theme(" /LIGHT "))
        self.assertEqual(contextual2.ui_theme, "light")
        self.assertFalse(contextual2.set_ui_theme("/light extra"))
        self.assertEqual(contextual2.ui_theme, "light")
        self.assertTrue(contextual2.set_ui_theme("/dark"))
        self.assertEqual(contextual2.ui_theme, "dark")

    def test_rich_startup_and_prompt_colours_follow_theme(self):
        for name in ("dark", "light"):
            contextual2.set_ui_theme("/" + name)
            colors = contextual2.UI_THEMES[name]
            self.assertEqual(str(contextual2.console.get_style("bold green")),
                             "bold " + colors["accent"])
            self.assertEqual(str(contextual2.console.get_style("white")), colors["input_fg"])
            self.assertEqual(str(contextual2.console.get_style("yellow")), colors["status"])

    def test_stream_preview_stays_inside_resized_terminal(self):
        content = Text("\n".join(f"line {i}: " + "wrapped text " * 15 for i in range(80)) + "\nEND")
        for width, height in ((80, 24), (35, 10), (100, 40), (20, 3)):
            console = Console(width=width, height=height)
            lines = console.render_lines(contextual2.StreamingPreview(content), pad=False)
            _, rendered_height = Segment.get_shape(lines)
            self.assertLessEqual(rendered_height, height - 2)
            self.assertIn("END", "".join(segment.text for line in lines for segment in line))

    @patch("contextual2.get_multiline_input", side_effect=["Hello", "/exit"])
    def test_completed_long_response_is_printed_once(self, prompt):
        answer = "\n".join(f"Unique response line {i}" for i in range(100))
        chunks = [{"message": {"content": line + "\n"}, "done": False} for line in answer.splitlines()]
        chunks.append({"message": {"content": ""}, "done": True})
        output = io.StringIO()
        console = Console(file=output, width=80, height=10)
        with patch.object(contextual2, "console", console), patch.object(contextual2, "stream_chat", return_value=iter(chunks)):
            contextual2.handle_chat_interaction(contextual2.ChatModel("ollama", "test-model"), False, {})
        self.assertEqual(output.getvalue().count("Unique response line 0"), 1)
        self.assertEqual(output.getvalue().count("Unique response line 99"), 1)

    def test_preview_is_cleared_before_live_stops_on_error(self):
        with patch.object(contextual2, "Live") as live_class:
            live = live_class.return_value.__enter__.return_value
            with self.assertRaises(RuntimeError):
                with contextual2.streaming_display() as display:
                    display.update(Text("unfinished response"))
                    raise RuntimeError("backend disconnected")
            self.assertEqual(live.update.call_args.args[0].plain, "")

    @patch("contextual2.get_multiline_input", side_effect=["Hello", "Again", "/exit"])
    def test_status_uses_last_turn_for_context_and_totals_for_usage(self, prompt):
        streams = [iter([{"message": {"content": "OK"}, "done": True,
                         "prompt_eval_count": 100, "eval_count": 20}]),
                   iter([{"message": {"content": "OK"}, "done": True,
                          "prompt_eval_count": 200, "eval_count": 30}])]
        with patch.object(contextual2, "stream_chat", side_effect=streams):
            contextual2.handle_chat_interaction(contextual2.ChatModel("afm", "system"), False, {})
        status = prompt.call_args.kwargs["status"]
        self.assertIn("Context 4,096", status)
        self.assertIn("~94.4% left", status)  # (4096 - 230) / 4096, not session usage.
        self.assertIn("In 300 / Out 50", status)

    def test_unknown_context_does_not_invent_a_percentage(self):
        status = contextual2.chat_status("model", "Chat", "Ready", None, 200, 100, 100)
        self.assertIn("Context unknown · n/a left", status)

    def test_stream_and_markdown_use_active_foreground_without_body_background(self):
        for theme in ("light", "dark", "light", "dark"):
            contextual2.set_ui_theme("/" + theme)
            colors = contextual2.UI_THEMES[theme]
            renderables = [contextual2.StreamingPreview(Text("Visible body text", style=colors["input_fg"])),
                           contextual2.themed_markdown("Visible body text\n\n## Heading\n\n- List item")]
            for renderable in renderables:
                lines = contextual2.console.render_lines(renderable, pad=False)
                segments = [segment for line in lines for segment in line if "Visible body text" in segment.text]
                self.assertTrue(segments)
                for segment in segments:
                    self.assertEqual(segment.style.color.name, colors["input_fg"])
                    self.assertIsNone(segment.style.bgcolor)
            self.assertIsNone(contextual2.console.get_style(contextual2.console.style).bgcolor)

    def test_theme_redraw_preserves_content_and_rethemes_previous_response(self):
        transcript = [{"prompt": "Question", "model": "Example model", "answer": "Original response"}]
        for theme in ("light", "dark"):
            contextual2.set_ui_theme("/" + theme)
            with patch.object(contextual2.console, "clear") as clear, patch.object(contextual2.console, "print") as output:
                contextual2.redraw_conversation(transcript)
                clear.assert_called_once()
                self.assertEqual(output.call_count, 2)
                rendered = output.call_args.args[0]
                lines = contextual2.console.render_lines(rendered, pad=False)
                segments = [s for line in lines for s in line if "Original response" in s.text]
                self.assertEqual(segments[0].style.color.name, contextual2.UI_THEMES[theme]["input_fg"])

    @patch("contextual2.stream_chat")
    @patch("contextual2.get_multiline_input", side_effect=["/light", "Hello", "/dark", "/exit"])
    def test_theme_commands_do_not_reach_model(self, prompt, stream):
        stream.return_value = iter([
            {"message": {"content": "**Hello**"}, "done": True},
        ])
        model = contextual2.ChatModel("ollama", "test-model")
        self.assertTrue(contextual2.handle_chat_interaction(model, False, {}))
        stream.assert_called_once()
        self.assertEqual(stream.call_args.args[1], [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "**Hello**"},
        ])
        self.assertIn("Response", prompt.call_args_list[-1].kwargs["status"])


if __name__ == "__main__":
    unittest.main()
