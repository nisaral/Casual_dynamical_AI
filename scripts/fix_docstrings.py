# Fix triple-double-quote docstrings inside code(r"""...""") builders.
from pathlib import Path


def fix_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    out = []
    i = 0
    marker = 'code(r"""'
    while True:
        j = text.find(marker, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        start = j + len(marker)
        k = start
        while True:
            k = text.find('"""', k)
            if k < 0:
                raise RuntimeError(f"Unclosed code block in {path}")
            after = text[k + 3 :].lstrip()
            if after.startswith(")"):
                content = text[start:k]
                content = content.replace('"""', "'''")
                out.append(marker)
                out.append(content)
                out.append('"""')
                i = k + 3
                break
            k = k + 3
    path.write_text("".join(out), encoding="utf-8")
    print(f"fixed {path.name}")


if __name__ == "__main__":
    root = Path(__file__).parent
    for p in root.glob("build_*.py"):
        fix_file(p)
