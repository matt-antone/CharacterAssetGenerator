import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from cag import cli
from cag.chat_codex import ChatCodex
from cag.chat_session import ChatSession, TextPending


def test_a_text_step_with_no_reply_writes_its_request_and_stops(tmp_path):
    model = ChatSession(folder=tmp_path)
    with pytest.raises(TextPending) as stop:
        model.invoke([SystemMessage("You direct."), HumanMessage("Direct the dance.")])
    request = stop.value.request
    assert request.read_text().startswith("You direct.")
    assert "User: Direct the dance." in request.read_text()
    assert not stop.value.answer.exists()
    assert str(request) in str(stop.value) and str(stop.value.answer) in str(stop.value)


def test_the_reply_is_read_back_once_written(tmp_path):
    model = ChatSession(folder=tmp_path)
    with pytest.raises(TextPending) as stop:
        model.invoke("Direct the dance.")
    stop.value.answer.write_text("  Mic in the character-right hand.\n")
    assert model.invoke("Direct the dance.").content == "Mic in the character-right hand."


def test_a_changed_prompt_asks_again_instead_of_reusing_a_reply(tmp_path):
    model = ChatSession(folder=tmp_path)
    with pytest.raises(TextPending) as first:
        model.invoke("Direct the dance.")
    first.value.answer.write_text("An answer to the first question.")
    with pytest.raises(TextPending) as second:
        model.invoke("Direct the victory.")
    assert second.value.answer != first.value.answer


def test_an_empty_reply_is_still_waiting(tmp_path):
    model = ChatSession(folder=tmp_path)
    with pytest.raises(TextPending) as stop:
        model.invoke("Direct the dance.")
    stop.value.answer.write_text("\n")
    with pytest.raises(TextPending):
        model.invoke("Direct the dance.")


def test_the_session_writes_the_text_unless_codex_is_asked_for(tmp_path, monkeypatch):
    monkeypatch.delenv("CAG_TEXT_MODEL", raising=False)
    model = cli.text_model(tmp_path)
    assert isinstance(model, ChatSession) and model.folder == tmp_path / "text"
    monkeypatch.setenv("CAG_TEXT_MODEL", "codex")
    assert isinstance(cli.text_model(tmp_path), ChatCodex)
