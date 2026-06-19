# scripts/

One-off research scripts. Anything exploratory (a render sweep, a prompt
experiment, a calibration plot) goes here, not in `src/`.

Run them with the project's environment:

```bash
uv run python scripts/my_experiment.py
```

These scripts are **not** part of the installed package. Don't import from
`scripts/` into `src/gaussian_robot/`.
