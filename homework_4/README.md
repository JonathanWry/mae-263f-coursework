# Homework 4: discrete elastic shells

This homework checks triangle-mesh connectivity, one shell hinge, and the
finite-difference gradient of its bending energy.

## Files

- `Homework4.py`: numerical data for Questions 1--8.
- `validate_homework4.py`: optional checks of the mesh counts, hinge angle,
  bending energy, and finite-difference gradient.

Questions 9--11 are conceptual time-integration questions and do not require
additional code.

The scripts import `getTheta`, `gradEb_hessEb_Shell`, and
`getEdgesAndHinges` from the course repository. Keep `mae-263F` beside this
coursework repository.

## Run

From the coursework repository root:

```bash
python3 homework_4/Homework4.py
```

Optional validation:

```bash
python3 homework_4/validate_homework4.py
```

The main script prints each question separately and reports numerical values;
it does not print the multiple-choice letters.
