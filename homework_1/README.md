# Homework 1 simulator

This single-file implementation follows the element-assembly structure from the
Homework 1 handout. Node 0 is the fixed wall, and element `i` connects nodes
`i - 1` and `i`.

## Files

- `Homework1.py`: element functions, residual/Jacobian assembly, implicit and
  explicit Euler methods, and simulations for Questions 1--11.
- `validate_homework1.py`: optional function tests and comparisons with the
  analytical solutions in the homework appendix.

## Run

From the `mae-263f-coursework` repository root:

```bash
python3 homework_1/Homework1.py
```

The script prints the simulated final position and velocity of every free node
for each homework question.
Optional validation can be run separately.
