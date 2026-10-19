"""Optional numerical and analytical checks for Homework2.py."""

import numpy as np

from Homework2 import (
    assemble_elastic,
    beam_length,
    euler_bernoulli_deflection,
    falling_terminal_velocity,
    simulate_falling_beam,
    simulate_supported_beam,
    supported_beam_parameters,
)
from elastic_derivatives import (
    bending_gradient_hessian,
    stretching_gradient_hessian,
)


def _finite_difference_jacobian(function, point, step=1.0e-6):
    """Central-difference Jacobian of a vector-valued function."""

    point = np.asarray(point, dtype=float)
    jacobian = np.empty((function(point).size, point.size))
    for column in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[column] = step
        jacobian[:, column] = (
            function(point + perturbation) - function(point - perturbation)
        ) / (2.0 * step)
    return jacobian


def validate_element_derivatives():
    """Compare the analytical element Hessians with finite differences."""

    stretching_q = np.array([0.1, -0.2, 0.45, 0.05])
    stretching_gradient, stretching_hessian = stretching_gradient_hessian(
        stretching_q, 0.4, 123.0
    )
    stretching_fd = _finite_difference_jacobian(
        lambda q: stretching_gradient_hessian(q, 0.4, 123.0)[0],
        stretching_q,
    )
    assert np.allclose(stretching_hessian, stretching_fd, rtol=1.0e-8, atol=1.0e-8)
    assert np.allclose(stretching_gradient.sum(), 0.0)

    bending_q = np.array([0.0, 0.0, 0.45, 0.03, 0.91, -0.02])
    _, bending_hessian = bending_gradient_hessian(
        bending_q, 0.0, 0.45, 2.7
    )
    bending_fd = _finite_difference_jacobian(
        lambda q: bending_gradient_hessian(q, 0.0, 0.45, 2.7)[0],
        bending_q,
        step=1.0e-7,
    )
    assert np.allclose(bending_hessian, bending_fd, rtol=2.0e-6, atol=2.0e-6)

    straight_q = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    full_gradient, _ = assemble_elastic(straight_q, 2.7, 123.0, 0.5)
    assert np.allclose(full_gradient, 0.0)
    print("Element derivative checks: passed")


def validate_part1():
    """Check the falling-beam results against Appendices A-C."""

    _, _, velocity_free_fall, _ = simulate_falling_beam(
        21, 0.01, 1.0, visc=0.0, R_all=0.025
    )
    expected_free_fall = -9.8 * (1.0 - 1000.0 / 7000.0)
    assert np.isclose(velocity_free_fall[-1], expected_free_fall, atol=1.0e-10)

    _, _, velocity_sphere, _ = simulate_falling_beam(
        21, 0.01, 1.0, visc=1000.0, R_all=0.025
    )
    expected_sphere = -2.0 * 0.025**2 * 6000.0 * 9.8 / (9.0 * 1000.0)
    assert np.isclose(velocity_sphere[-1], expected_sphere, atol=1.0e-10)

    # The unequal-radius beam needs several seconds to finish changing shape.
    _, _, velocity_beam, _ = simulate_falling_beam(21, 0.01, 10.0)
    expected_beam = falling_terminal_velocity(21)
    assert np.isclose(velocity_beam[-1], expected_beam, rtol=5.0e-5)
    print("Part 1 analytical checks: passed")


def validate_part2():
    """Check geometry, boundary conditions, and representative loads."""

    parameters = supported_beam_parameters()
    assert np.isclose(parameters["delta_length"], 1.0 / 49.0)
    assert np.isclose(parameters["EI"], 765.2919704144734)
    assert np.isclose(parameters["EA"], 1.0555751316061705e7)
    assert np.isclose(parameters["lumped_mass"], 8.309191998066064e-3)
    assert parameters["loaded_node"] == 37
    assert 2 * parameters["loaded_node"] + 1 == 75
    assert parameters["free_indices"].size == 97

    _, minimum_y_small, q_small, iterations_small = simulate_supported_beam(2000.0)
    assert np.isclose(abs(minimum_y_small[-1]), 0.0371092385, rtol=2.0e-6)
    assert np.allclose(q_small[parameters["fixed_indices"]], [0.0, 0.0, 0.0])
    assert iterations_small <= 10
    assert np.isclose(euler_bernoulli_deflection(2000.0), 0.0380449156)

    _, minimum_y_large, q_large, iterations_large = simulate_supported_beam(20000.0)
    assert np.isclose(abs(minimum_y_large[-1]), 0.235256258, rtol=2.0e-6)
    assert iterations_large <= 10
    assert q_large[-2] < 1.0
    assert beam_length(q_large) - 1.0 < 4.0e-4
    print("Part 2 geometry and simulation checks: passed")


def main():
    validate_element_derivatives()
    validate_part1()
    validate_part2()
    print("All Homework 2 validations passed.")


if __name__ == "__main__":
    main()
