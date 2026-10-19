# MAE 263F Coursework

Personal coursework repository for UCLA MAE 263F, *Mechanics of Flexible
Structures and Soft Robots*.

## Assignments

- [`homework_1/`](homework_1/): modular one-dimensional
  mass-spring-damper simulator using implicit and explicit Euler integration.
- [`homework_2/`](homework_2/): implicit planar-beam simulator for a beam falling
  in viscous fluid and a point-loaded simply supported beam.

Each assignment directory contains its own source code, run instructions, and
optional validation utilities. Course handouts and lecture files are not
redistributed in this repository.

## Environment

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
