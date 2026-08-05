"""Execute notebooks in their own directories (so savfig paths work)."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Probabilistic_Graphs_and_State" / "Inference_Algorithms_and_Causality.ipynb",
    ROOT / "State_Space_Models" / "SSM_S4_Mamba_HiPPO.ipynb",
    ROOT / "Reasoning_and_Planning" / "MCTS_Lookahead_Planning.ipynb",
    ROOT / "World_Models" / "Dreamer_JEPA_World_Models.ipynb",
    ROOT / "Meta_Learning" / "MAML_and_Fast_Adaptation.ipynb",
]


def run_one(path: Path) -> None:
    print(f"\n=== Executing {path.relative_to(ROOT)} ===")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except Exception as e:
        print(f"FAILED: {e}")
        # still write partial for debugging
        nbformat.write(nb, path)
        raise
    nbformat.write(nb, path)
    print(f"OK: {path.name}")


if __name__ == "__main__":
    for p in NOTEBOOKS:
        run_one(p)
    print("\nAll notebooks executed.")
