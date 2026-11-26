# Bow-string-arrow simulation

The main model is `distributed_bow_archery.py`.

- The rigid riser is fixed.
- Each bow limb is a multi-segment planar DER with distributed mass,
  stretching, bending, and a prescribed unstrung shape.
- The massive string is a tension-only planar chain attached to the two moving
  limb tips. A very small bending term regularizes its slack shape but is not
  used as physical bowstring stiffness.
- The arrow is a rigid translating mass.
- One-sided penalty contact transfers force from the string to the arrow and
  allows separation without prescribing a release position.
- Backward Euler and Newton iteration advance the coupled dynamics.

The earlier `coupled_archery.py` single-hinge limb model is retained only as a
baseline. Its entire limb can rotate through the brace plane, so its
post-release shape should not be interpreted as a physical bow shape.

The default case is a straight-end approximation calibrated against the
Greenhorn Comet TD 350 measurements reported by Tuijn and Kooi (1992). A full
working-recurve model is intentionally excluded because it would also require
frictionless string--limb contact and a moving string detachment point.

## Run

```bash
python3 final_project/distributed_bow_archery.py
```

Animation:

```bash
python3 final_project/distributed_bow_archery.py --animate
```

Save an MP4 instead of opening a window:

```bash
python3 final_project/distributed_bow_archery.py --no-show \
  --save-animation final_project/distributed_bow_launch.mp4
```

Verification:

```bash
python3 final_project/validate_distributed_bow.py
```

The verification checks the assembled derivatives, zero energy of the natural
limb shape, the tension-only cable law, convergence to Euler--Bernoulli
cantilever theory, force--draw work versus state energy, static residuals,
unilateral contact and release, fixed root clamps, distributed limb rotation,
published bow data, and the configured draw-force and strain constraints. Run
the slower mesh/time-step, contact-stiffness, and string-regularization
sensitivity checks with:

```bash
python3 final_project/validate_distributed_bow.py --extended
```

Default result (10 limb segments and 21 string nodes): 0.207 m brace height,
130.4 N full-draw force, 30.8 J available draw energy, 49.27 m/s release speed,
and 71.35% efficiency. The corresponding published values are 0.214--0.220 m,
126.9 N, 32.16 J, 49.73 m/s, and 70.2%. The simulated limb tips remain behind
the riser after release.

## Model sources

- B. W. Kooi, "On the mechanics of the bow and arrow," *Journal of
  Engineering Mathematics*, 1981, https://doi.org/10.1007/BF00052515.
- C. Tuijn and B. W. Kooi, "The measurement of arrow velocities in the
  students' laboratory," *European Journal of Physics*, 1992,
  https://www.bio.vu.nl/thb/users/kooi/tuko92.pdf.
- M. Tiermas, "The limb deformation of the compound bow," *Meccanica*, 2017,
  https://doi.org/10.1007/s11012-016-0485-0.
