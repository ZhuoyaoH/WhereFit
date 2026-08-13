# Project Layout

This repository uses a Python `src` layout. The source package is `src/wherefit/`; the root-level `WhereFit/` directory is not application source code.

## Tracked project files

```text
.
├── app.py                 # Streamlit entrypoint
├── src/wherefit/          # Application package
│   ├── data_sources/      # Weather and hazard data providers
│   ├── hazards/           # Hazard and aurora summaries
│   ├── report/            # Chinese report generation
│   ├── scoring/           # Scoring model
│   └── visualization/     # Charts, cards, maps
├── data/
│   ├── city_seed.csv      # City metadata and explicit emergency fallback levels
│   ├── climate/           # Versioned 2000-2025 NASA POWER city aggregates and manifest
│   └── air_quality/       # Versioned 2015-2024 ACAG PM2.5 extraction and manifest
├── scripts/               # Reproducible public-data builders
├── tests/                 # Unit tests
├── docs/                  # Repository documentation
├── README.md
├── requirements-dev.txt
├── requirements.txt
└── pytest.ini
```

## Local-only directories

These directories are generated or local to this machine and should not be treated as source code:

```text
WhereFit/                # Local runtime folder, ignored by Git
├── .conda-env/          # Existing local conda environment
├── .conda-pkgs/         # Local conda package cache
└── .pip-cache/          # Local pip cache

data/cache/              # API/cache and resumable builder state regenerated locally
FromChatgpt/             # Local planning/source exports
改进计划/                 # Local planning notes
```

## Why keep `src/wherefit/`

`src/wherefit/` is the importable Python package. This is a standard Python project layout that prevents accidental imports from the repository root and keeps tests closer to how installed code behaves.

## Why not move `WhereFit/.conda-env`

The current conda environment contains scripts whose shebangs point to the absolute path:

```text
/Users/zhuoyao/Documents/WhereFit/WhereFit/.conda-env
```

Moving that directory would preserve files but can break entrypoints such as `streamlit`. For now, the safer cleanup is to document `WhereFit/` as local runtime state and keep it ignored. New environments should use a less confusing path such as `.conda-env/` at the repo root or an external conda/mamba environment outside the repository.
