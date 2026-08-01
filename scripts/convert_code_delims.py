from pathlib import Path


def convert_code_blocks(text: str) -> str:
    """Convert code(r\"\"\"...\"\"\") to code(r'''...''')."""
    marker = 'code(r"""'
    closer = '""")'
    out = []
    i = 0
    while True:
        j = text.find(marker, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        start = j + len(marker)
        # Find the LAST-safe closer: first """) that appears at the beginning of a line
        # (builders always close code cells that way). Fall back to first """) .
        k = -1
        pos = start
        while True:
            pos = text.find(closer, pos)
            if pos < 0:
                break
            # check if only whitespace before closer on this line
            line_start = text.rfind("\n", start, pos) + 1
            prefix = text[line_start:pos]
            if prefix.strip() == "":
                k = pos
                break
            pos += 3
        if k < 0:
            k = text.find(closer, start)
        if k < 0:
            raise RuntimeError("Unclosed code(r\"\"\") block")
        content = text[start:k]
        out.append("code(r'''")
        out.append(content)
        out.append("''')")
        i = k + len(closer)
    return "".join(out)


if __name__ == "__main__":
    root = Path(__file__).parent
    for name in [
        "build_inference_notebook.py",
        "build_ssm_notebook.py",
        "build_mcts_notebook.py",
    ]:
        p = root / name
        text = p.read_text(encoding="utf-8")
        new = convert_code_blocks(text)
        p.write_text(new, encoding="utf-8")
        print(f"converted {name}")
