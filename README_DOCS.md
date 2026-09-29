# ClimAID documentation (0.3.0, under testing)

Source: `mkdocs.yml`, `site_docs/`, `overrides/` (site-wide "under testing" banner).
Built site: `site/` (open `site/index.html` via a local server, or publish the folder).

Build:
    pip install -r requirements-docs.txt
    mkdocs build          # run from the repository root, next to the climaid/ package

The API pages are generated from the package docstrings, so build from the repository root.
Stay on MkDocs 1.x: MkDocs 2.0 removes plugins and theme overrides.
