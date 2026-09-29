#!/usr/bin/env sh
# Build the documentation and bundle it inside the ClimAID package (climaid/documentation),
# so `climaid docs` and the dashboard's /documentation pages work offline.
# Run from the repository root (next to mkdocs.yml and the climaid/ package).
set -e
mkdocs build -d site
rm -rf climaid/documentation
cp -r site climaid/documentation
echo "Documentation bundled in climaid/documentation"
