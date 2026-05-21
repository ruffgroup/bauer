"""Execute a notebook cell-by-cell with per-cell progress output.

Each completed cell is saved immediately (so progress persists across restarts)
and a one-line status is printed — designed to be consumed by the Monitor tool.
"""
import sys
import time
import nbformat
from nbclient import NotebookClient


def run(path: str, timeout: int = 7200) -> int:
    nb = nbformat.read(path, as_version=4)
    total = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"START {path} ({total} code cells, per-cell timeout {timeout}s)", flush=True)

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()
    try:
        done = 0
        t0 = time.time()
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            done += 1
            t_cell = time.time()
            src_preview = "".join(cell.source).splitlines()[0][:70] if cell.source else ""
            try:
                client.execute_cell(cell, index)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL cell {done}/{total} ({time.time()-t_cell:.1f}s): {exc!s}", flush=True)
                nbformat.write(nb, path)
                return 1
            nbformat.write(nb, path)
            print(
                f"OK cell {done}/{total} "
                f"({time.time()-t_cell:.1f}s, total {time.time()-t0:.0f}s) "
                f"| {src_preview}",
                flush=True,
            )
        print(f"DONE {path} in {time.time()-t0:.0f}s", flush=True)
        return 0
    finally:
        try:
            client._cleanup_kernel()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
