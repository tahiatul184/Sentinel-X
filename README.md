# Sentinel-X

Sentinel-X is an open-source satellite awareness and monitoring project.

## Quick start

1. Clone this repository.
2. Install Python dependencies from the requirements files.
3. Run the dashboard:

```bash
python start_dashboard.py
```

## Project structure

- `app/` - core modules, services, pipelines, and tests
- `start_dashboard.py` - dashboard launcher
- `desktop.py` - desktop interface launcher

## Requirements

Python 3.x is required. See:

- `app/requirements-foundation.txt`
- `app/requirements-dashboard.txt`

## Testing

Run the included test files with pytest after installing dependencies.

## Notes

This repository is provided for experimentation and testing. Configure any required external services or credentials before running collection modules.
