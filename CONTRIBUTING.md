# Contributing to IndexPulse

Thanks for your interest in contributing! Here is how to get started.

## Development Setup

```bash
git clone https://github.com/officethree/IndexPulse.git
cd IndexPulse
python -m venv .venv
source .venv/bin/activate
make install
```

## Running Tests

```bash
make test          # basic test run
make test-cov      # with coverage report
```

## Code Quality

Before submitting a pull request, make sure linting and formatting pass:

```bash
make lint          # ruff + black --check
make format        # auto-fix formatting
make typecheck     # mypy strict mode
```

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure all checks pass (`make lint && make test`).
4. Keep commits focused — one logical change per commit.
5. Write a clear PR description explaining *why* the change is needed.

## Reporting Issues

Open an issue on GitHub with:
- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behaviour
- Python version and OS

## Code of Conduct

Be kind, be constructive, be respectful.
