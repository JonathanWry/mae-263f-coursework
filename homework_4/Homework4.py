"""Numerical data for MAE 263F Homework 4, Questions 1--8."""

from pathlib import Path
import sys

import numpy as np


COURSE_REPOSITORY = Path(__file__).resolve().parents[2] / "mae-263F"
sys.path.insert(0, str(COURSE_REPOSITORY / "appendix"))
sys.path.insert(0, str(COURSE_REPOSITORY / "src" / "shells"))

from gradEb_hessEb_Shell import (  # noqa: E402
    getTheta,
    gradEb_hessEb_Shell,
)
from shell_functions import getEdgesAndHinges  # noqa: E402


def plate_elements() -> np.ndarray:
    """Return the eight triangles of the 3-by-3-node plate in Figure 1."""
    elements = []
    for j in range(2):
        for i in range(2):
            lower_left = 3 * j + i
            lower_right = lower_left + 1
            upper_left = lower_left + 3
            upper_right = upper_left + 1
            elements.append([lower_left, lower_right, upper_right])
            elements.append([lower_left, upper_right, upper_left])
    return np.asarray(elements, dtype=int)


def bending_energy(
    q: np.ndarray, theta_bar: float = 0.0, bending_stiffness: float = 1.0
) -> float:
    """Return Eb = 1/2 kb (theta - theta_bar)^2 for one hinge."""
    theta = getTheta(q)
    return float(0.5 * bending_stiffness * (theta - theta_bar) ** 2)


def perturbed_energies(
    q: np.ndarray, epsilon: float = 0.01
) -> np.ndarray:
    """Return the 12 one-sided perturbed energies used in Questions 3--7."""
    energies = np.zeros(12)
    for coordinate in range(12):
        perturbed_q = q.copy()
        perturbed_q[coordinate] += epsilon
        energies[coordinate] = bending_energy(perturbed_q)
    return energies


def finite_difference_gradient(
    q: np.ndarray, epsilon: float = 0.01
) -> np.ndarray:
    return (perturbed_energies(q, epsilon) - bending_energy(q)) / epsilon


def question_header(number: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"Q{number}: {title}")


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    elements = plate_elements()
    edges, hinges = getEdgesAndHinges(elements)
    question_header(1, "edges and hinges")
    print(f"number of edges  = {len(edges)}")
    print(f"number of hinges = {len(hinges)}")
    print("edges:")
    print(edges)
    print("hinges:")
    print(hinges)

    q = np.array(
        [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
            0.5, -1.0, 1.0,
        ]
    )
    theta = float(getTheta(q))
    original_energy = bending_energy(q)
    question_header(2, "hinge angle and bending energy")
    print(f"theta [rad] = {theta:.12f}")
    print(f"Eb [J]      = {original_energy:.12f}")

    energies = perturbed_energies(q)
    for node in range(4):
        question_header(node + 3, f"perturbing node x{node}")
        print("Eb after x, y, z perturbations [J]:")
        print(energies[3 * node : 3 * node + 3])

    gradient_fd = finite_difference_gradient(q)
    question_header(7, "finite-difference gradient")
    print("gradFD [N], grouped as [x0 | x1 | x2 | x3]:")
    print(gradient_fd.reshape(4, 3))

    analytical_gradient, _ = gradEb_hessEb_Shell(
        q, None, None, None, 0.0, 1.0
    )
    question_header(8, "analytical gradient from course code")
    print("dF [N], grouped as [x0 | x1 | x2 | x3]:")
    print(analytical_gradient.reshape(4, 3))
    print(
        "maximum |gradFD - dF| [N] = "
        f"{np.max(np.abs(gradient_fd - analytical_gradient)):.8f}"
    )


if __name__ == "__main__":
    main()
