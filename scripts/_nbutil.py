"""Helpers to build notebooks without triple-quote hell."""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf


def new_nb():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = []
    return nb


def add_md(nb, text: str):
    nb["cells"].append(nbf.v4.new_markdown_cell(text.strip("\n")))


def add_code(nb, text: str):
    nb["cells"].append(nbf.v4.new_code_cell(text.strip("\n")))


def save(nb, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {path}")
