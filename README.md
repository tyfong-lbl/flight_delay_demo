# Flight Delay Demo

## Environment Bootstrap

Use the bootstrap script to create a local virtual environment, install dependencies,
and validate required imports.

```bash
python3 scripts/bootstrap_env.py --log-level INFO
```

For import validation only (without pip install), run:

```bash
python3 scripts/bootstrap_env.py --skip-install --log-level DEBUG
```

The dependency list is defined in `requirements.txt` and follows the project spec.
