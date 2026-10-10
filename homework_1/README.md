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
for each homework question. It also prints derived quantities requested by the
oscillator questions, such as mechanical energy, energy ratio, and interval maximum.
It does not print or hard-code multiple-choice selections. Install the packages
in `requirements.txt` if NumPy is not available.

Questions 9--11 also produce position-time plots in `homework_1/figures`. Q9 overlays
both time-step cases on the same axes, Q10 shows the full explicit-Euler run,
and Q11 marks the simulated maximum over the requested 0--10 s interval. Each
plot is saved as PDF, SVG, and PNG.

Optional validation can be run separately:

```bash
python3 homework_1/validate_homework1.py
```

For a different problem, call `simulate` with parameter arrays that include the
fixed wall entry:

```python
params = make_params(
    m=[0.0, m1, m2, m3],
    k=[0.0, k1, k2, k3],
    b=[0.0, b1, b2, b3],
    g=9.8,
    dt=0.01,
)
```

Choose `method="implicit"` or `method="explicit"` in the call to `simulate`.
