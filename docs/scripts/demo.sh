#!/usr/bin/env bash
# Regenerate docs/assets/demo.png from a real run of docs/scripts/demo.py.
# Needs a built venv (uv venv && uvx maturin develop --uv) and freeze
# (https://github.com/charmbracelet/freeze). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."

freeze --execute ".venv/bin/python docs/scripts/demo.py" \
  --theme github-dark --background "#0d1117" --window --padding 24 \
  --font.family "Menlo" --font.size 28 --output docs/assets/demo.png
