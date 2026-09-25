"""A LangChain chat model whose replies are written by the session's own AI.

cag runs inside an agent session — Claude Code, Codex, whichever is driving the
build — and that agent is already a capable writer. Shelling out to a second
model for the text steps (the motion director, a written motion sheet, a bible
a brief did not assemble itself) spent a separate subscription, and when that
subscription hit its usage cap every set failed before a picture was drawn.

So a text step here does not call anything. It writes the prompt it would have
sent to `<folder>/<key>.prompt.md` and stops with `TextPending`, the way the key
art stops for approval. The session's AI reads the request, writes its reply to
`<folder>/<key>.md`, and builds again; the reply is then read back as the
model's answer. `key` is a digest of the prompt, so a reply is reused only for
the exact question it answered, and a changed brief asks afresh.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .chat_codex import render_prompt


class TextPending(RuntimeError):
    """Raised when a text step has no reply yet. `answer` is where to write it."""

    def __init__(self, answer: Path):
        self.answer = answer
        self.request = answer.with_suffix(".prompt.md")
        super().__init__(
            f"waiting for text: answer {self.request} by writing the reply to {self.answer}, "
            "then build again"
        )


class ChatSession(BaseChatModel):
    """Chat model answered through files by the agent session running cag."""

    folder: Path

    @property
    def _llm_type(self) -> str:
        return "session"

    def paths(self, prompt: str) -> tuple[Path, Path]:
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return self.folder / f"{key}.prompt.md", self.folder / f"{key}.md"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = render_prompt(messages)
        request, answer = self.paths(prompt)
        text = answer.read_text().strip() if answer.exists() else ""
        if not text:
            request.parent.mkdir(parents=True, exist_ok=True)
            request.write_text(prompt + "\n")
            raise TextPending(answer)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
