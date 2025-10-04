#!/usr/bin/env bash
set -euo pipefail

# core dirs
mkdir -p src/{api,features,training,inference}
mkdir -p data/{raw,interim,processed}
mkdir -p models/exoplanet_stack_v1

# package inits
touch src/__init__.py
touch src/api/__init__.py
touch src/features/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py

# keep empty data/model dirs tracked
touch data/raw/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch models/exoplanet_stack_v1/.gitkeep

# helper files (only if absent)
[ -e .gitignore ] || cat > .gitignore <<'GIT'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
# Data/Models (keep only .gitkeep)
data/*
!data/**/.gitkeep
models/*
!models/**/.gitkeep
# Misc
.DS_Store
GIT

[ -e README.md ] || cat > README.md <<'MD'
# NYXION-BACKEND

Backend scaffold for the exoplanet ML project.

## Folders
- `src/`      : Python packages (api/features/training/inference)
- `data/`     : raw/interim/processed datasets
- `models/`   : versioned model artifacts
MD

# placeholder entry point (if absent)
[ -e main.py ] || printf '"""\nEntry point (FastAPI or runner) — empty scaffold\n"""\n' > main.py
