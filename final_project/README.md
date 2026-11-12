# Soft bowstring and rigid-arrow prediction

`archery_prediction.py` is the main final project. It predicts the launch and
flight of a rigid arrow driven by a deformable bowstring.

## Model

| Component | Representation |
|---|---|
| Bow | Rigid support with two fixed string endpoints |
| Bowstring | Discrete soft body with stretching energy |
| Arrow | Rigid translating body; its mass is coupled to the center nock before release |
| Release | Arrow detaches when the string reaches the brace plane |
| Flight | Gravity and quadratic aerodynamic drag |

The string uses the stretching gradient and Hessian from
`homework_2/elastic_derivatives.py`. Each launch step is solved with backward Euler
and Newton iteration, following the numerical structure used in the course
homework. A string has negligible bending stiffness, so this model intentionally
uses stretching energy rather than DER rod bending or twist energy.

The effective axial stiffness `EA` is calibrated from the specified full-draw
force. For half-span `h`, draw distance `d`, and half-string length
`l = sqrt(h^2 + d^2)`,

```text
strain = (l - h) / h
F_draw = 2 EA strain d / l.
```

Before release, the rigid arrow mass is added only to the nock's axial degree of
freedom. When the nock reaches `x = 0`, the arrow separates with the nock velocity.
That predicted velocity becomes the initial condition of a rigid-body flight model.
The program then solves for the low aim angle that reaches the requested target
height and distance.

The default model uses 21 string nodes, a `0.70 m` draw, a `200 N` full-draw force,
a `25 g` arrow, and a 70 m target. These are effective demonstration parameters and
can be changed from the command line.

## Default prediction

The current default result is approximately:

- initial string energy: `42.40 J`;
- release time: `17.2 ms`;
- release speed: `55.35 m/s`;
- energy-only upper limit: `58.24 m/s`;
- launch energy efficiency: about `90%`;
- required aim angle for a 70 m equal-height target: `6.70 deg`;
- time to target: `1.305 s`.

## Run

From the coursework repository root:

```bash
python3 final_project/archery_prediction.py --animate
```

The temporary figures show the string shape, arrow acceleration and nock force,
energy transfer, and predicted arrow trajectory. No figures are saved.

Useful parameters:

```bash
python3 final_project/archery_prediction.py \
  --nodes 41 \
  --draw-force 220 \
  --draw-length 0.72 \
  --target-distance 70 \
  --animate
```

Run the checks with:

```bash
python3 final_project/validate_archery.py
```

The validation checks the calibrated draw force, energy bound, 11/21/41-node mesh
convergence, and target-height prediction.

## Scope and limitations

The soft string is an effective representation of the energy-storing bow-string
system. A real bow stores most of its energy in bending limbs, and a real arrow can
flex, rotate, and experience lift. Those effects are deliberately excluded here so
the project remains within the course scope: a soft elastic structure coupled to a
rigid body, followed by a simple aerodynamic prediction.

The earlier soft-ball files remain in this directory as development prototypes,
but they are not the main final-project model.
