import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cag.chat_codex import ChatCodex, CodexError, render_prompt


def test_render_prompt_puts_system_first_and_labels_turns():
    prompt = render_prompt([
        SystemMessage("You draw."),
        HumanMessage("Draw a cat."),
        AIMessage("Done."),
        HumanMessage("Again."),
    ])
    assert prompt.startswith("You draw.")
    assert "User: Draw a cat." in prompt
    assert "Assistant: Done." in prompt
    assert prompt.endswith("Assistant:")


def test_command_flags():
    argv = ChatCodex(model="gpt-5.3-codex")._command(Path("/tmp/out.txt"))
    assert argv[:3] == ["codex", "exec", "--ephemeral"]
    assert "--ignore-user-config" in argv
    assert argv[-3:] == ["--model", "gpt-5.3-codex", "-"]
    assert "--output-last-message" in argv


def test_generate_reads_last_message(monkeypatch):
    def fake_run(argv, **kwargs):
        Path(argv[argv.index("--output-last-message") + 1]).write_text("  hello  ")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ChatCodex().invoke("hi").content == "hello"


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "boom")
    )
    with pytest.raises(CodexError, match="boom"):
        ChatCodex().invoke("hi")


def test_generate_raises_on_empty_message(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", "")
    )
    with pytest.raises(CodexError, match="no final message"):
        ChatCodex().invoke("hi")
