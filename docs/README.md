# bauer documentation

The docs are built with **Sphinx** and published to **GitHub Pages** automatically:
the `.github/workflows/docs.yml` Action builds them and deploys to the `gh-pages`
branch **on every push to `main`** — no manual step required. Live site:
<https://ruffgroup.github.io/bauer/>. (A `.readthedocs.yaml` is also present as an
optional alternative host; see [Alternative hosting: ReadTheDocs](#alternative-hosting-readthedocs).)

Tutorial pages are Jupyter notebooks that are pre-executed locally and committed
with outputs; Sphinx renders them via `nbsphinx` without re-executing.

**TL;DR**
- *Build locally:* `cd docs && make html` (or `make docs` from the repo root).
- *Publish:* `git push origin main` — the Action rebuilds and deploys to GitHub Pages.

---

## Building locally

```bash
conda activate bauer          # or: pip install -e ".[docs]"
cd docs
make html                     # output in _build/html/
open _build/html/index.html   # macOS quick-view
```

---

## Publishing

**You don't push the built HTML — you push the source.** The GitHub Action
`.github/workflows/docs.yml` rebuilds the docs and deploys them to the `gh-pages`
branch on every push to `main`:

```bash
git push origin main          # → Action builds & deploys to GitHub Pages
```

Watch the deploy under the repo's **Actions** tab (the "Build and deploy docs" run),
or `gh run watch`. The live site is <https://ruffgroup.github.io/bauer/>; it updates
a minute or two after the Action finishes.

Notes:
- The Action only **deploys** on `main`; on PRs it builds (to catch errors) but
  does not publish.
- Notebooks are **not** re-executed by the Action — it renders the committed
  `.ipynb` outputs. Re-execute them locally (see [Editing tutorials](#editing-tutorials))
  and commit the updated `.ipynb` before pushing.

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
│   └── lesson{1..9}.ipynb # Pre-executed notebooks (source: make_notebooks.py)
└── requirements.txt      # Sphinx + nbsphinx + nbconvert (for ReadTheDocs)
```

---

## Editing tutorials

**Do not edit the `.ipynb` files directly.** They are generated and executed
from a single source file:

```
docs/tutorial/make_notebooks.py
```

### Quick workflow: full rebuild

Run this script from the repo root after any change to `make_notebooks.py`:

```bash
conda activate bauer
./docs/tutorial/build_tutorials.sh
```

This does three things in sequence:
1. Regenerates all `.ipynb` files from `make_notebooks.py`
2. Executes each notebook in order (can take 1–3 h for MCMC lessons)
3. Rebuilds the Sphinx HTML and verifies outputs appear

Then commit and push:

```bash
git add docs/tutorial/*.ipynb
git commit -m "Re-execute tutorial notebooks"
git push origin main
```

### Skip re-execution (fast check)

If you only changed Sphinx config or `.rst` files and don't need to re-run the
notebooks, pass `--no-exec`:

```bash
./docs/tutorial/build_tutorials.sh --no-exec
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

## Alternative hosting: ReadTheDocs

The live site is **GitHub Pages** (see [Publishing](#publishing)). ReadTheDocs is
*not* required, but the repo ships a `.readthedocs.yaml` so it can be hosted there
too if desired. The config:
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
| `furo` | HTML theme (set in `conf.py`) |
| `nbsphinx` | Renders Jupyter notebooks |
| `nbconvert` | Executes notebooks locally |
| `nbformat` | Reads/writes `.ipynb` files |
