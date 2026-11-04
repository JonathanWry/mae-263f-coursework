# Homework 3 simulator

This implementation simulates straight and naturally curved clamped-free rods
with the Discrete Elastic Rods method, implicit Euler, and Newton's method.

## Files

- `Homework3.py`: simulations and numerical outputs for the computational
  questions.
- `validate_homework3.py`: optional analytical and convergence checks.
- `src/der/der_functions.py`: course-provided DER helper functions.
- `appendix/`: course-provided elastic-energy derivatives.

## Run

From the `mae-263f-coursework` repository root:

```bash
python3 homework_3/Homework3.py
```

The script prints the numerical results and displays temporary plots for
Questions 4, 10, and 11. Optional validation can be run separately:

```bash
python3 homework_3/validate_homework3.py
```
