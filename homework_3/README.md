# Homework 3 — Discrete Elastic Rods

This folder solves the straight and naturally curved clamped-free rod cases in
Homework 3. `Homework3.py` runs the computational questions, prints general
numerical data and convergence diagnostics, and does not choose multiple-choice
answers. Purely conceptual Questions 3 and 8 are not implemented in code.

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 homework_3/Homework3.py
```

The full parameter sweeps contain several three-dimensional nonlinear rod runs
and can take several minutes. Per-step plotting and printing are disabled. At
the end, temporary figures show the Q4 mesh convergence, the Q10 twist-angle
comparison, and the Q11 shape and end-distance evolution; the figures are not
saved to disk.

## Files

- `Homework3.py`: new homework-specific geometry, mass and gravity setup;
  implicit-Euler/Newton stepping; steady-state checks; Euler–Bernoulli check;
  and the Q1–Q11 runner.
- `validate_homework3.py`: analytical-formula, geometry, mesh-convergence,
  final-force-balance, symmetry, and natural-curvature checks.
- `src/der/der_functions.py`: course-provided DER frame, curvature, elastic-force,
  and Jacobian assembly functions, copied without functional changes from
  `mae-263F/src/der/der_functions.py`.
- `appendix/*.py`: course-provided analytical stretching, bending, and twisting
  derivatives and their small geometry helpers, copied from
  `mae-263F/appendix/`.

The copied helper code identifies its original author and license in
`src/der/der_functions.py`. The new homework-specific changes are confined to
`Homework3.py`.

## Main interface

```python
run_rod(
    nv=52,
    length=0.2,
    youngs_modulus=10e6,
    gravity=9.81,
    natural_radius=0.0,
    dt=0.1,
    total_time=10.0,
    remove_natural_curvature=False,
)
```

It returns a `RodResult` containing the final DOF vector `q`, tip-
displacement and end-distance histories, final twist angles, reference edge
lengths, Newton-iteration information, final static residual, and the relative
change over the final second. The straight-rod geometry uses `L/(N-2)` so that
the last `N-2` edges, not the clamped first edge, have total free length `L`.

Run the optional independent checks with:

```bash
python3 homework_3/validate_homework3.py
```
