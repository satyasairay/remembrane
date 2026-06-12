# Contributing to remembrane

Thanks for your interest!

## Setup

```bash
git clone https://github.com/satyasairay/remembrane
cd remembrane
pip install -e .[dev]
pytest
```

## Guidelines

- Keep the core dependency-free. New integrations go in `src/remembrane/adapters/` as duck-typed, optional modules.
- Every PR needs tests. Run `pytest` and `ruff check src tests` before submitting.
- One feature per PR. Open an issue first for anything large.

## Ideas welcome

Good first contributions: new framework adapters, embedder backends, recall strategies (MMR, hybrid keyword+vector), benchmarks.
