"""Checks for the Homework 4 mesh, energy, and gradient calculations."""

import numpy as np

from Homework4 import (
    bending_energy,
    finite_difference_gradient,
    getEdgesAndHinges,
    getTheta,
    gradEb_hessEb_Shell,
    plate_elements,
)


def main() -> None:
    edges, hinges = getEdgesAndHinges(plate_elements())
    assert len(edges) == 16
    assert len(hinges) == 8

    q = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         0.5, 1.0, 0.0, 0.5, -1.0, 1.0]
    )
    np.testing.assert_allclose(getTheta(q), -np.pi / 4.0)
    np.testing.assert_allclose(bending_energy(q), np.pi**2 / 32.0)

    gradient_fd = finite_difference_gradient(q)
    analytical_gradient, _ = gradEb_hessEb_Shell(
        q, None, None, None, 0.0, 1.0
    )
    assert np.max(np.abs(gradient_fd - analytical_gradient)) < 0.005

    print("Plate edge and hinge counts: passed")
    print("Hinge angle and energy: passed")
    print("Finite-difference gradient check: passed")


if __name__ == "__main__":
    main()
