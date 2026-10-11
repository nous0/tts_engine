"""Tests for the podcast CLI's script reading (examples/generate_podcast.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.core import podcast as podcast_core

_PATH = Path(__file__).resolve().parents[1] / "examples" / "generate_podcast.py"
_spec = importlib.util.spec_from_file_location("generate_podcast", _PATH)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _turns(path: Path) -> list[tuple[str, str]]:
    data = cli.read_script(path)
    return [(t.speaker, t.text) for t in podcast_core.parse_script(data)]


def test_reads_a_plain_text_transcript(tmp_path):
    path = tmp_path / "show.txt"
    path.write_text("Alice: Hi Bob.\nBob: Hi Alice!\n", encoding="utf-8")
    assert _turns(path) == [("Alice", "Hi Bob."), ("Bob", "Hi Alice!")]


def test_reads_a_json_list(tmp_path):
    path = tmp_path / "show.json"
    path.write_text(json.dumps([{"speaker": "A", "text": "One."}]), encoding="utf-8")
    assert _turns(path) == [("A", "One.")]


def test_reads_a_written_script_with_genders_and_title(tmp_path):
    path = tmp_path / "topic.script.json"
    body = {
        "title": "Ep",
        "turns": [{"speaker": "Lan", "text": "Hello."}],
        "genders": {"Lan": "female"},
        "normalize_text": True,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    data = cli.read_script(path)
    assert data["title"] == "Ep"
    assert data["genders"] == {"Lan": "female"}
    assert _turns(path) == [("Lan", "Hello.")]


def test_reads_the_repo_examples():
    examples = _PATH.parent
    assert len(_turns(examples / "script_trio.txt")) == 6
    assert _turns(examples / "script.json")
