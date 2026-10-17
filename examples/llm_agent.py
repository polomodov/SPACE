#!/usr/bin/env python3
"""A minimal LLM agent: Claude writes shell commands, SPACE runs them in a microVM.

Claude receives a task and a single `run_command` tool. Every tool call becomes a
SPACE command session in one Firecracker sandbox; its exit code, stdout, and
stderr go back to Claude until it returns a final answer. The sandbox is deleted
at the end unless --keep is given.

Requires the `anthropic` package and an Anthropic credential (ANTHROPIC_API_KEY
or an `ant auth login` profile), plus the same SPACE_* variables and local setup
as examples/first_sandbox.py, whose HTTP helpers this script reuses:
    pip install anthropic
    python3 examples/llm_agent.py
    python3 examples/llm_agent.py "Count the lines in /etc/passwd"
    python3 examples/llm_agent.py --keep "Write hello.txt and show it"
"""

import argparse
import base64
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import first_sandbox as space  # noqa: E402


MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
# On a safety-classifier refusal the API retries on Anthropic's recommended model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TURNS = 12
COMMAND_TIMEOUT = 60
OUTPUT_LIMIT = 8000
DEFAULT_TASK = (
    "Report the OS release, kernel version, CPU count and memory of this sandbox, "
    "then compute the sum of all primes below 100000 with any tool available inside it."
)

SYSTEM = f"""You solve the user's task by running shell commands in an isolated Linux \
sandbox (a Firecracker microVM) through the run_command tool.

Each call starts a fresh `sh -c` process in the home directory: files persist between \
calls, but the working directory and shell variables do not. Commands are killed after \
{COMMAND_TIMEOUT} seconds. Check which interpreters exist before relying on one: the \
minimal seed template has no Python, while the code-runtime template has Python and \
Node.js. To run a longer program, write it to a file with a heredoc, then execute it.

Verify results by running them. When you are done, reply with the answer and one or \
two sentences on how you got it."""

TOOLS = [{
    "name": "run_command",
    "description": (
        "Run a shell script with `sh -c` inside the sandbox and return its exit code, "
        "stdout and stderr. Long output is truncated in the middle."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell script passed to `sh -c`."},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}]


def clip(data):
    text = data.decode("utf-8", errors="replace")
    if len(text) <= OUTPUT_LIMIT:
        return text
    half = OUTPUT_LIMIT // 2
    return f"{text[:half]}\n... [{len(text) - OUTPUT_LIMIT} characters omitted] ...\n{text[-half:]}"


def run_in_sandbox(config, directory, state, script, name):
    """Run one command session and collect its output; nonzero exits are results, not errors."""
    sandbox_id = space.require_id(state, "sandbox_id", "sbx_")
    body = {"command": ["sh", "-c", script], "timeout": f"{COMMAND_TIMEOUT}s"}
    command = space.mutation(config, directory, state, name, "POST",
                             f"/v1/sandboxes/{sandbox_id}/commands", body)
    session_id = space.require_id(command, "session_id", "proc_")
    attach = command.get("attach")
    if not isinstance(attach, dict):
        raise ValueError("Command response has no attachment")
    events_path = f"/v1/sessions/{session_id}/events"
    events_url = config["data_plane_url"] + events_path
    if attach.get("events_url") not in {events_path, events_url}:
        raise ValueError("Unexpected events URL; do not bypass the destination check")
    token = attach.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("Command attachment token is missing")

    stdout, stderr = io.BytesIO(), io.BytesIO()
    exited, exit_code = False, None
    with tempfile.NamedTemporaryFile(prefix="events-", suffix=".sse", dir=directory, delete=False) as transcript:
        state["transcripts"].append(Path(transcript.name).name)
        space.save(directory, state)
        request = Request(events_url, headers={
            "Authorization": f"Bearer {token}", "Accept": "text/event-stream",
        })
        stream_timeout = COMMAND_TIMEOUT + 15
        with space.deadline(stream_timeout):
            with space.HTTP.open(request, timeout=stream_timeout) as response:
                for frame in space.event_frames(response, transcript):
                    kind = frame.get("type")
                    if kind in {"error", "resync_required"}:
                        raise RuntimeError(f"Command stream reported {kind}; keep the transcript")
                    if kind in {"stdout", "stderr"}:
                        output = stdout if kind == "stdout" else stderr
                        output.write(base64.b64decode(frame["data_base64"], validate=True))
                    if kind == "process_exited":
                        exited, exit_code = True, frame.get("exit_code")
    if not exited:
        raise RuntimeError("Stream ended without a process_exited event")
    return exit_code, stdout.getvalue(), stderr.getvalue()


def format_result(exit_code, stdout, stderr):
    return f"exit_code: {exit_code}\n--- stdout ---\n{clip(stdout)}\n--- stderr ---\n{clip(stderr)}"


def run_agent(client, task, execute, log=print):
    """The agent loop: ask Claude, run each requested command, return the results."""
    messages = [{"role": "user", "content": task}]
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined the task")
        # Append the full content, thinking blocks included, so history stays append-only.
        messages.append({"role": "assistant", "content": response.content})
        text = "".join(block.text for block in response.content if block.type == "text")
        if text:
            log(f"Claude: {text}", flush=True)
        if response.stop_reason == "end_turn":
            return text
        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unexpected stop reason: {response.stop_reason}")

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            command = block.input.get("command") if isinstance(block.input, dict) else None
            if block.name != "run_command" or not isinstance(command, str) or not command.strip():
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": "Expected run_command with a non-empty command."})
                continue
            log(f"$ {command}", flush=True)
            result = execute(command)
            log(result, flush=True)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"No final answer within {MAX_TURNS} turns")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--keep", action="store_true", help="keep the sandbox for inspection")
    args = parser.parse_args(argv)
    directory = state = config = None
    try:
        config = space.configuration()
        try:
            import anthropic
        except ImportError:
            raise ValueError("Install the Anthropic SDK first: pip install anthropic") from None
        client = anthropic.Anthropic()
        directory = Path(tempfile.mkdtemp(prefix="space-llm-agent-"))
        state = {"api_url": config["api_url"], "data_plane_url": config["data_plane_url"],
                 "requests": {}, "transcripts": []}
        space.save(directory, state)
        print(f"Run directory: {directory}", flush=True)
        space.create_sandbox(config, directory, state)
        names = (f"command-{index}" for index in itertools.count(1))
        answer = run_agent(client, args.task, lambda script: format_result(
            *run_in_sandbox(config, directory, state, script, next(names))))
        print(f"\nAnswer:\n{answer}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Stopped: {error or 'interrupted'}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if state and state.get("sandbox_id"):
            if args.keep:
                print(f"Sandbox kept. Delete it with: python3 examples/first_sandbox.py delete {directory}")
            else:
                try:
                    space.delete_sandbox(config, directory, state)
                except Exception as error:
                    print(f"Cleanup failed: {error}. Retry with: "
                          f"python3 examples/first_sandbox.py delete {directory}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
