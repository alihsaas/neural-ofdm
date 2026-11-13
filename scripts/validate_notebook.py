import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute an OFDM notebook in a fresh project kernel"
    )
    parser.add_argument("path", nargs="?", default="notebooks/01_classical_ofdm_baseline.ipynb")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".runtime"
    kernel_dir = runtime / "kernels" / "neural-ofdm"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "NeuralOFDM validation",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    manager = KernelManager(
        kernel_name="neural-ofdm",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]),
    )
    path = (root / args.path).resolve()
    if not path.is_relative_to(root / "notebooks") or path.suffix != ".ipynb":
        parser.error("path must be a project notebook")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=180,
        resources={"metadata": {"path": str(root / "notebooks")}},
    )
    print("Executing all cells in a fresh project kernel...", flush=True)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
    executed = sum(
        cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells
    )
    print(f"Validated {executed} code cells; saved outputs to {path}")


if __name__ == "__main__":
    main()
