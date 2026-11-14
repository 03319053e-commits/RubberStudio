"""Every translation key used in the code exists in both languages, with matching placeholders."""
import ast
import re
import string
from pathlib import Path

from rubberstudio.strings import STRINGS

ROOT = Path(__file__).resolve().parents[1] / "rubberstudio"
KEY = re.compile(r"^(ai|bt|canvas|crop|dlg|er|file|fold|hdr|help|hist|q|rep|splash|st|tool|wm)(\.[a-z0-9_]+)+$")
DYNAMIC = [f"help.{k}.{part}" for k in ("crop", "repair", "watermark", "erase", "batch", "keys") for part in "hb"]
DYNAMIC += [f"rep.hint.{k}" for k in ("heal", "smudge", "texture")]


def _used_keys() -> set:
    keys = {"busy"}
    for f in ROOT.rglob("*.py"):
        if f.name == "strings.py":
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and KEY.match(node.value):
                keys.add(node.value)
    return keys | set(DYNAMIC)


def _fields(s: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(s) if name}


def test_same_keys_in_both_languages():
    en, zh = set(STRINGS["en"]), set(STRINGS["zh"])
    assert en == zh, (sorted(en - zh), sorted(zh - en))


def test_all_used_keys_are_defined():
    missing = [k for k in sorted(_used_keys()) if k not in STRINGS["en"]]
    assert not missing, missing


def test_no_unused_keys_pile_up():
    used = _used_keys()
    unused = sorted(k for k in STRINGS["en"] if k not in used)
    assert len(unused) <= 6, unused           # a few spare keys are fine, a graveyard is not


def test_placeholders_match():
    for k, en in STRINGS["en"].items():
        assert _fields(en) == _fields(STRINGS["zh"][k]), k
