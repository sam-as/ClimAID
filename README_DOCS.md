# Building and releasing the ClimAID documentation

| Path | Role | In git |
|---|---|---|
| `mkdocs.yml` | Site configuration | yes |
| `site_docs/` | Documentation **source** (Markdown) | yes |
| `overrides/` | Theme overrides: site-wide "under testing" banner, footer with NDMC logo | yes |
| `docs_hooks.py` | MkDocs hook: reads the version from `climaid/__init__.py` for the banner | yes |
| `CHANGELOG.md` | Included verbatim in the site's Changelog page | yes |
| `climaid/documentation/` | Built site **bundled in the package** (`climaid docs`, dashboard `/documentation/`) | yes |
| `site/` | Local build output, published to GitHub Pages | no (ignored) |

## Build

```bash
pip install -r requirements-docs.txt
./build_and_bundle.sh      # mkdocs build --strict, then copy site/ to climaid/documentation/
```

Run from the repository root (next to `mkdocs.yml` and the `climaid/` package): the API pages are generated
from the package docstrings. Stay on MkDocs 1.x; MkDocs 2.0 removes plugins and theme overrides.

## Releasing a new version

1. Change the version in **one place**: `__version__` in `climaid/__init__.py`. `pyproject.toml`
   reads it from there, as do the dashboard and the documentation banner.
2. Update the prose that names the version: the new section in `CHANGELOG.md`, the "What's new" section of
   `README.md`, and the version mentioned in `site_docs/index.md` and `site_docs/guide/status.md`.
3. Rebuild and re-bundle the documentation with `./build_and_bundle.sh`.
4. Run `pytest -m "not slow"`. `tests/test_version_consistency.py` fails if any of the above still names
   another version, if the bundled documentation is stale, or if a file has a byte-order mark or garbled
   characters.
5. Commit, publish the website with `mkdocs gh-deploy`, then push a tag `vX.Y.Z` (matching `__version__`).
   The *Publish to PyPI* workflow builds and uploads the package for that tag (it also runs when a GitHub
   release is published; do one or the other, as PyPI refuses a second upload of the same version).

## Editing on Windows

Save files as **UTF-8 without BOM**. PowerShell 5 `Set-Content`/`Out-File` write a BOM and can garble characters
such as `—` and `–` (each turns into a run of odd characters starting with `â`); use an editor, or PowerShell 7 with `-Encoding utf8NoBOM`.
`.editorconfig` and `.gitattributes` in the repository set the expected encoding and line endings.
