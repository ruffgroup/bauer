# bauer documentation

The docs are built with **Sphinx** and hosted on **ReadTheDocs**.
Tutorial pages are Jupyter notebooks that are pre-executed locally and committed
with outputs; Sphinx renders them via `nbsphinx` without re-executing.

---

## Building locally

```bash
conda activate bauer          # or: pip install -e ".[docs]"
cd docs
make html                     # output in _build/html/
open _build/html/index.html   # macOS quick-view
```

---

## Doc structure

```
docs/
├── conf.py               # Sphinx config (theme, extensions, version)
├── index.rst             # Top-level toctree
├── installation.rst      # Installation guide
├── concepts.rst          # Conceptual overview of models
├── api_reference.rst     # Auto-generated API docs (autodoc)
├── tutorial/
│   ├── index.rst         # Tutorial toctree (add new lessons here)
│   ├── make_notebooks.py # Single source of truth for all tutorial notebooks
│   ├── lesson1.ipynb     # Pre-executed notebook — psychophysics / NLC
│   ├── lesson2.ipynb     # Pre-executed notebook — KLW / Barreto-García 2023
│   ├── lesson3.ipynb     # Pre-executed notebook — PMCM / de Hollander 2024
│   └── lesson4.ipynb     # Pre-executed notebook — Flexible noise model
└── requirements.txt      # Sphinx + nbsphinx + nbconvert (for ReadTheDocs)
```

---

## Editing tutorials

**Do not edit the `.ipynb` files directly.** They are generated and executed
from a single source file:

```
docs/tutorial/make_notebooks.py
```

Workflow for changing a tutorial:

```bash
cd docs/tutorial
conda activate bauer

# 1. Edit make_notebooks.py
#    Each lesson is a list of md() and code() cells.

# 2. Regenerate the .ipynb files
python make_notebooks.py

# 3. Execute the notebook(s) you changed
#    (nbsphinx renders with nbsphinx_execute='never', so outputs must be committed)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    lesson1.ipynb --output lesson1.ipynb

# 4. Rebuild docs to check rendering
cd .. && make html
```

For lessons with MCMC sampling (lessons 2–4), execution can take 1–3 hours.
Run them in the background:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
    lesson2.ipynb --output lesson2.ipynb &
```

### Adding a new lesson

1. Add a new `nb5 = nbf.v4.new_notebook()` block in `make_notebooks.py`
2. Write the cells using the `md()` / `code()` helpers
3. Write the notebook with `nbf.write(nb5, 'lesson5.ipynb')`
4. Add `lesson5.ipynb` to `tutorial/index.rst`
5. Execute and commit

---

## Adding or editing reference pages

Plain `.rst` files in `docs/` are standard Sphinx reStructuredText.
To add a new page:

1. Create `docs/mypage.rst`
2. Add it to the toctree in `docs/index.rst`

The API reference (`api_reference.rst`) is generated automatically from
docstrings via `sphinx.ext.autodoc`.  Update docstrings in the source code;
the docs will pick them up on the next build.

---

## ReadTheDocs

The repo contains `.readthedocs.yaml` which configures the build:
- Python 3.11, installs `pip install -e ".[docs]"`
- Runs `sphinx-build` on `docs/conf.py`
- Does **not** re-execute notebooks (`nbsphinx_execute = 'never'` in `conf.py`)

To enable ReadTheDocs hosting:
1. Push the repo to GitHub (e.g. `ruffgroup/bauer`)
2. Go to [readthedocs.org](https://readthedocs.org), import the project
3. Point it at the GitHub repo — it picks up `.readthedocs.yaml` automatically
4. Trigger a build; pre-executed notebooks render immediately

---

## Dependencies

Runtime dependencies for building docs (also in `pyproject.toml [docs]`):

| Package | Purpose |
|---------|---------|
| `sphinx` | Core doc builder |
| `sphinx-rtd-theme` | ReadTheDocs theme |
| `nbsphinx` | Renders Jupyter notebooks |
| `nbconvert` | Executes notebooks locally |
| `nbformat` | Reads/writes `.ipynb` files |
