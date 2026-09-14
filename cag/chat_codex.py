"""A LangChain chat model backed by the local `codex` CLI.

The ChatGPT subscription is the only model access this project has; the OpenAI
API is out of scope. `codex exec` is therefore the transport: it takes a prompt,
runs a turn, and writes the final assistant message to a file.

Text only. No streaming, no tool calling — the graph decides what happens next,
not the model. Add `bind_tools` if a node ever actually needs it.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ROLE_LABELS = {"human": "User", "ai": "Assistant", "tool": "Tool"}


class CodexError(RuntimeError):
    """Raised when the codex CLI fails or returns nothing."""


def render_prompt(messages: list[BaseMessage]) -> str:
    """Flatten a message list into one prompt.

    System messages lead, as instructions. The rest becomes a labelled
    transcript ending in an empty Assistant turn for the model to fill.
    """
    system = [str(m.content) for m in messages if m.type == "system"]
    turns = [
        f"{ROLE_LABELS.get(m.type, m.type.capitalize())}: {m.content}"
        for m in messages
        if m.type != "system"
    ]
    parts = ["\n\n".join(system)] if system else []
    parts.append("\n\n".join(turns))
    parts.append("Assistant:")
    return "\n\n".join(p for p in parts if p)


class ChatCodex(BaseChatModel):
    """Chat model that shells out to `codex exec`."""

    model: str | None = None
    timeout: int = 600
    #: Sandbox for shell commands the model may run. Reading is enough for
    #: prompt-authoring roles; image generation uses its own tool, not this.
    sandbox: str = "read-only"
    cwd: str | None = None
    #: Skip ~/.codex/config.toml so the user's own model, hooks and agent
    #: instructions cannot leak into pipeline turns.
    ignore_user_config: bool = True

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def _command(self, out_file: Path) -> list[str]:
        argv = [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            self.sandbox,
            "--output-last-message",
            str(out_file),
        ]
        if self.ignore_user_config:
            argv.append("--ignore-user-config")
        if self.model:
            argv += ["--model", self.model]
        if self.cwd:
            argv += ["--cd", self.cwd]
        argv.append("-")  # prompt arrives on stdin
        return argv

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = render_prompt(messages)
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "last-message.txt"
            try:
                result = subprocess.run(
                    self._command(out_file),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexError(f"codex exec timed out after {self.timeout}s") from exc
            if result.returncode != 0:
                raise CodexError(
                    f"codex exec failed ({result.returncode}): {result.stderr.strip()[-2000:]}"
                )
            text = out_file.read_text() if out_file.exists() else ""
        if not text.strip():
            raise CodexError("codex exec produced no final message")
        message = AIMessage(content=text.strip())
        return ChatResult(generations=[ChatGeneration(message=message)])
