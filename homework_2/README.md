# Homework 2 beam simulators

This implementation uses one elastic-force assembly and one implicit-Euler/Newton
solver for both parts of the assignment. Question cases are created by changing
only the relevant simulator arguments; simulations shared by multiple questions
are run once and reused.

## Files

- `Homework2.py`: falling-beam and simply supported-beam simulators, numerical
  data for Questions 1-10, and comparison plots for the relevant questions.
- `elastic_derivatives.py`: local gradients and Hessians for the stretching and
  bending energies used by the beam assembly.
- `validate_homework2.py`: optional derivative, analytical, geometry, boundary-
  condition, and representative simulation checks.

## Run

From the `mae-263f-coursework` repository root, install the dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 homework_2/Homework2.py
```

The main script prints each question in its own section; it does not print
multiple-choice selections or decide which statement is correct. Questions 5,
7, 8, and 10 display temporary plots. The figures are not saved to disk. The
Question 10 plot shows the percentage relative difference `e(P)`, and Question
11 prints the large-load endpoint and stretching-spring diagnostics suggested
in the handout.

Optional validation:

```bash
python3 homework_2/validate_homework2.py
```

## Minimal question changes

Part 1 uses:

```python
simulate_falling_beam(
    nv,
    dt,
    total_time,
    visc=1000.0,
    R_all=None,
)
```

- Questions 1-2 set `R_all=0.025` and change only `visc`.
- Questions 3-4 share the default `N=21`, 50-second run.
- Question 5 changes only `nv`; the default small radius remains `deltaL/10`.

Part 2 uses:

```python
simulate_supported_beam(load, nv=50, dt=0.01, total_time=1.0)
```

Questions 6-10 therefore change only `load`. The `P=2000 N` result is reused
for Questions 6-8, and the `P=20000 N` result is reused in the load sweep.
