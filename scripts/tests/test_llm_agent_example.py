"""Offline checks for the LLM agent example: a fake SPACE server and a scripted Claude client."""

import base64
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_sandbox_example import server  # noqa: E402

SOURCE = Path(__file__).resolve().parents[2] / "examples" / "llm_agent.py"
spec = importlib.util.spec_from_file_location("llm_agent_example", SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def sse(*frames):
    return b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)


def output(kind, text):
    return {"type": kind, "data_base64": base64.b64encode(text.encode()).decode()}


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(command, block_id="toolu_1", name="run_command"):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input={"command": command})


class ScriptedClaude:
    """Returns the queued responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        stop_reason, content = self.responses.pop(0)
        return SimpleNamespace(stop_reason=stop_reason, content=content)


class SandboxRoutes:
    """A fake control plane and data plane for one sandbox."""

    def __init__(self, events=None):
        self.events = events or sse(output("stdout", "hi\n"), {"type": "process_exited", "exit_code": 0})
        self.commands = []

    def __call__(self, method, path, body):
        json_type = {"Content-Type": "application/json"}
        if method == "POST" and path == "/v1/sandboxes":
            return 200, json_type, {"sandbox_id": "sbx_test", "operation_id": "op_create"}
        if method == "GET" and path.startswith("/v1/operations/"):
            return 200, json_type, {"state": "succeeded"}
        if method == "GET" and path == "/v1/sandboxes/sbx_test":
            return 200, json_type, {"state": "running"}
        if method == "GET" and path.startswith("/v1/sandboxes/sbx_test/processes"):
            return 200, json_type, {"data": []}
        if method == "POST" and path == "/v1/sandboxes/sbx_test/commands":
            self.commands.append(json.loads(body))
            session = f"proc_{len(self.commands)}"
            return 200, json_type, {"session_id": session, "process_id": session, "attach": {
                "token": "attachment-secret", "events_url": f"/v1/sessions/{session}/events"}}
        if method == "GET" and path.startswith("/v1/sessions/"):
            return 200, {"Content-Type": "text/event-stream"}, self.events
        if method == "DELETE" and path == "/v1/sandboxes/sbx_test":
            return 200, json_type, {"operation_id": "op_delete"}
        return 404, json_type, {"error": "not found"}


class LlmAgentExampleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(self.stderr))

    def sandbox(self, routes):
        origin, calls = self.stack.enter_context(server(routes))
        config = {"api_url": origin, "data_plane_url": origin, "token": "management-secret",
                  "template": "seed:latest"}
        state = {"api_url": origin, "data_plane_url": origin, "requests": {}, "transcripts": [],
                 "sandbox_id": "sbx_test"}
        return config, state, calls

    def test_command_output_and_nonzero_exit_are_returned(self):
        routes = SandboxRoutes(sse(output("stdout", "out\n"), output("stderr", "err\n"),
                                   {"type": "process_exited", "exit_code": 3}))
        config, state, calls = self.sandbox(routes)
        result = agent.run_in_sandbox(config, self.directory, state, "echo out; exit 3", "command-1")
        self.assertEqual(result, (3, b"out\n", b"err\n"))
        self.assertEqual(routes.commands, [{"command": ["sh", "-c", "echo out; exit 3"], "timeout": "60s"}])
        stream = [call for call in calls if call[1] == "/v1/sessions/proc_1/events"][0]
        self.assertEqual(stream[2]["Authorization"], "Bearer attachment-secret")
        self.assertNotIn("secret", (self.directory / "state.json").read_text())

    def test_stream_without_exit_or_with_error_fails(self):
        for events in (sse(output("stdout", "partial")), sse({"type": "error"})):
            with self.subTest(events=events):
                config, state, _ = self.sandbox(SandboxRoutes(events))
                with self.assertRaises(RuntimeError):
                    agent.run_in_sandbox(config, self.directory, state, "true", f"command-{id(events)}")

    def test_untrusted_events_url_never_receives_token(self):
        config = {"api_url": "http://127.0.0.1:8080", "data_plane_url": "http://127.0.0.1:8090",
                  "token": "management-secret", "template": "seed:latest"}
        state = {"sandbox_id": "sbx_test", "requests": {}, "transcripts": []}
        response = {"session_id": "proc_1", "attach": {
            "token": "attachment-secret", "events_url": "http://untrusted.invalid/v1/sessions/proc_1/events"}}
        with patch.object(agent.space, "api", return_value=response), \
                patch.object(agent.space.HTTP, "open") as opened:
            with self.assertRaisesRegex(ValueError, "events URL"):
                agent.run_in_sandbox(config, self.directory, state, "true", "command-1")
            opened.assert_not_called()

    def test_long_output_is_clipped_in_the_middle(self):
        clipped = agent.clip(b"a" * agent.OUTPUT_LIMIT + b"b" * 100)
        self.assertTrue(clipped.startswith("a") and clipped.endswith("b"))
        self.assertIn("[100 characters omitted]", clipped)

    def test_agent_runs_tool_calls_and_returns_final_answer(self):
        claude = ScriptedClaude(
            ("tool_use", [text("Checking."), tool_use("uname -r")]),
            ("end_turn", [text("Kernel 6.1.")]),
        )
        executed = []
        answer = agent.run_agent(claude, "Which kernel?", lambda c: executed.append(c) or "exit_code: 0")
        self.assertEqual(answer, "Kernel 6.1.")
        self.assertEqual(executed, ["uname -r"])
        first, second = claude.requests
        self.assertEqual(first["messages"], [{"role": "user", "content": "Which kernel?"}])
        self.assertEqual(first["fallbacks"], "default")
        self.assertEqual(first["betas"], [agent.FALLBACK_BETA])
        self.assertEqual(first["thinking"], {"type": "adaptive"})
        self.assertTrue(first["tools"][0]["strict"])
        self.assertEqual(second["messages"][2], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "exit_code: 0"}]})

    def test_invalid_tool_call_is_reported_without_running(self):
        claude = ScriptedClaude(
            ("tool_use", [tool_use("   ", "toolu_a"), tool_use("ls", "toolu_b", name="other")]),
            ("end_turn", [text("Done.")]),
        )
        agent.run_agent(claude, "task", lambda _: self.fail("must not execute"))
        results = claude.requests[1]["messages"][2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["toolu_a", "toolu_b"])
        self.assertTrue(all(r["is_error"] for r in results))

    def test_refusal_and_turn_limit_stop_the_agent(self):
        with self.assertRaisesRegex(RuntimeError, "declined"):
            agent.run_agent(ScriptedClaude(("refusal", [])), "task", lambda _: "")
        endless = ScriptedClaude(*[("tool_use", [tool_use("true")])] * agent.MAX_TURNS)
        with self.assertRaisesRegex(RuntimeError, "No final answer"):
            agent.run_agent(endless, "task", lambda _: "exit_code: 0", log=lambda *a, **k: None)

    def run_main(self, routes, claude, *argv):
        origin, calls = self.stack.enter_context(server(routes))
        fake_sdk = SimpleNamespace(Anthropic=lambda: claude)
        env = {"SPACE_API_TOKEN": "management-secret", "SPACE_API_URL": origin,
               "SPACE_DATA_PLANE_URL": origin}
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"anthropic": fake_sdk}), \
                patch.object(agent.tempfile, "mkdtemp", return_value=str(self.directory)):
            return agent.main(list(argv)), calls

    def test_main_creates_runs_and_deletes_the_sandbox(self):
        claude = ScriptedClaude(("tool_use", [tool_use("echo hi")]), ("end_turn", [text("It said hi.")]))
        code, calls = self.run_main(SandboxRoutes(), claude, "Say hi")
        self.assertEqual(code, 0)
        self.assertIn("exit_code: 0\n--- stdout ---\nhi\n", claude.requests[1]["messages"][2]["content"][0]["content"])
        self.assertIn(("DELETE", "/v1/sandboxes/sbx_test"), [(c[0], c[1]) for c in calls])
        self.assertTrue(json.loads((self.directory / "state.json").read_text())["deleted"])
        self.assertEqual(list(self.directory.glob("events-*.sse")), [])

    def test_main_deletes_the_sandbox_after_a_failure_unless_kept(self):
        code, calls = self.run_main(SandboxRoutes(), ScriptedClaude(("refusal", [])))
        self.assertEqual(code, 1)
        self.assertIn(("DELETE", "/v1/sandboxes/sbx_test"), [(c[0], c[1]) for c in calls])
        self.stack.close()
        self.setUp()
        code, calls = self.run_main(SandboxRoutes(), ScriptedClaude(("end_turn", [text("ok")])), "--keep")
        self.assertEqual(code, 0)
        self.assertNotIn("DELETE", [c[0] for c in calls])
        self.assertIn("first_sandbox.py delete", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
