"""Independent analytical, convergence, balance, and limit checks for HW3."""

import numpy as np

from Homework3 import (
    euler_bernoulli_deflection,
    run_rod,
)


def check_analytical_formula_and_geometry():
    """Check Eq. (2), the clamp edge, mass geometry, and curved chord length."""

    delta_eb = euler_bernoulli_deflection(10.0e6, 0.05)
    np.testing.assert_allclose(delta_eb, 0.004, rtol=1.0e-13)

    straight = run_rod(nv=52, gravity=0.05)
    np.testing.assert_allclose(straight.reference_lengths, 0.004, rtol=1.0e-13)
    np.testing.assert_allclose(np.sum(straight.reference_lengths[1:]), 0.2)
    np.testing.assert_allclose(np.sum(straight.reference_lengths), 0.204)
    assert straight.steady_relative_change < 1.0e-3
    assert straight.relative_static_residual < 1.0e-4

    curved = run_rod(nv=50, natural_radius=0.02, gravity=9.81)
    expected_discrete_length = 49 * 2 * 0.02 * np.sin((0.2 / 0.02 / 49) / 2)
    np.testing.assert_allclose(
        np.sum(curved.reference_lengths), expected_discrete_length, rtol=1.0e-13
    )
    assert curved.steady_relative_change < 1.0e-3
    assert curved.relative_static_residual < 1.0e-4
    print("Analytical, geometry, and final-force-balance checks: passed")
    return straight, curved


def check_mesh_convergence():
    """Verify that small-load DER converges to Euler-Bernoulli with refinement."""

    delta_eb = euler_bernoulli_deflection(10.0e6, 0.05)
    errors = []
    deflections = []
    for nv in (22, 42, 82):
        result = run_rod(nv=nv, gravity=0.05)
        delta_der = -result.tip_z[-1]
        deflections.append(delta_der)
        errors.append((delta_der - delta_eb) / delta_eb)
        assert result.steady_relative_change < 1.0e-3
        assert result.relative_static_residual < 1.0e-4

    errors = np.asarray(errors)
    assert np.all(errors > 0.0)
    assert np.all(np.diff(errors) < 0.0)
    ratios = errors[1:] / errors[:-1]
    assert np.all((ratios > 0.4) & (ratios < 0.6))
    print("Mesh-convergence check: passed")
    print("  deflections [mm] =", 1000.0 * np.asarray(deflections))
    print("  relative errors  =", errors)


def check_symmetry_and_natural_curvature(straight, curved):
    """Check the expected twist symmetry and the zero-curvature limiting case."""

    assert np.max(np.abs(straight.final_twist)) < 1.0e-12
    assert np.max(np.abs(curved.final_twist)) > 1.0e-2

    zero_curvature = run_rod(
        nv=50,
        natural_radius=0.02,
        gravity=0.0,
        remove_natural_curvature=True,
    )
    discrete_length = np.sum(zero_curvature.reference_lengths)
    assert zero_curvature.end_distance[-1] > 0.999 * discrete_length
    assert zero_curvature.steady_relative_change < 1.0e-3
    assert zero_curvature.relative_static_residual < 1.0e-4

    print("Symmetry and natural-curvature checks: passed")


def check_load_stiffness_scaling():
    """Verify identical static shapes when the governing ratio g/E is fixed."""

    soft = run_rod(youngs_modulus=10.0e6, gravity=0.981)
    stiff = run_rod(youngs_modulus=100.0e6, gravity=9.81)
    maximum_q_difference = np.max(np.abs(soft.q - stiff.q))
    assert maximum_q_difference < 1.0e-5
    print("Load/stiffness scaling check: passed")
    print(f"  maximum DOF difference = {maximum_q_difference:.9g}")


def main():
    straight, curved = check_analytical_formula_and_geometry()
    check_mesh_convergence()
    check_symmetry_and_natural_curvature(straight, curved)
    check_load_stiffness_scaling()
    print("All Homework 3 validations passed.")


if __name__ == "__main__":
    main()
