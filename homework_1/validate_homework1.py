"""Optional validation for Homework1.py.

Run this file only when you want to check the element functions and compare the
question simulations with the analytical formulas in the homework appendix.
"""

import numpy as np

from Homework1 import (
    dampingForce,
    dampingJacobian,
    explicitEulerStep,
    getFJ,
    gradEs,
    hessEs,
    implicitEulerStep,
    make_params,
    simulate,
)


def run_case(m, k, b, dt, total_time, method="implicit"):
    """Run a homework case after adding the fixed wall at index 0."""

    m = np.r_[0.0, np.asarray(m, dtype=float)]
    k = np.r_[0.0, np.asarray(k, dtype=float)]
    b = np.r_[0.0, np.asarray(b, dtype=float)]
    params = make_params(m, k, b, dt=dt, g=9.8, tol=1.0e-8)
    return simulate(
        np.zeros(m.size),
        np.zeros(m.size),
        params,
        total_time,
        method=method,
    )


def check_close(label, simulated, analytical, atol=1.0e-8):
    """Assert agreement and print the numerical error."""

    error = abs(simulated - analytical)
    if not np.isclose(simulated, analytical, rtol=1.0e-7, atol=atol):
        raise AssertionError(
            f"{label}: simulation {simulated} != analytical {analytical}"
        )
    print(f"{label}: passed (absolute error {error:.3e})")


def static_position(masses, stiffnesses, node, gravity=9.8):
    """Appendix Eq. (16)."""

    masses = np.asarray(masses, dtype=float)
    stiffnesses = np.asarray(stiffnesses, dtype=float)
    return sum(
        gravity * np.sum(masses[i:]) / stiffnesses[i]
        for i in range(node)
    )


def terminal_velocity(masses, dampers, node, gravity=9.8):
    """Appendix Eq. (15)."""

    masses = np.asarray(masses, dtype=float)
    dampers = np.asarray(dampers, dtype=float)
    return sum(
        gravity * np.sum(masses[i:]) / dampers[i]
        for i in range(node)
    )


def oscillator_energy(x, u, equilibrium, mass, stiffness):
    """Mechanical energy measured relative to the static equilibrium."""

    return 0.5 * mass * u**2 + 0.5 * stiffness * (x - equilibrium) ** 2


def validate_required_functions():
    """Run the function-level test cases supplied in the handout."""

    assert np.allclose(gradEs(0.0, 0.1, 10.0), [-1.0, 1.0])
    assert np.allclose(hessEs(10.0), [[10.0, -10.0], [-10.0, 10.0]])
    assert np.allclose(
        dampingForce(0.0, 0.01, 0.0, 0.0, 2.0, 0.01), [-2.0, 2.0]
    )
    assert np.allclose(
        dampingJacobian(2.0, 0.01),
        [[200.0, -200.0], [-200.0, 200.0]],
    )

    stiffness_params = make_params(
        m=[0.0, 0.0, 0.0, 0.0],
        k=[0.0, 1.0, 2.0, 3.0],
        b=[0.0, 0.0, 0.0, 0.0],
        g=0.0,
        dt=1.0,
    )
    zeros = np.zeros(4)
    _, jacobian = getFJ(zeros, zeros, zeros, stiffness_params)
    assert np.allclose(
        jacobian[1:, 1:],
        [[3.0, -2.0, 0.0], [-2.0, 5.0, -3.0], [0.0, -3.0, 3.0]],
    )

    full_params = make_params(
        m=[0.0, 1.0, 2.0, 3.0],
        k=[0.0, 10.0, 20.0, 30.0],
        b=[0.0, 1.0, 2.0, 3.0],
        g=9.8,
        dt=0.1,
    )
    x_old = np.zeros(4)
    u_old = np.array([0.0, 0.1, 0.2, 0.3])
    x_new = np.array([0.0, 0.01, 0.03, 0.06])
    residual, jacobian = getFJ(x_new, x_old, u_old, full_params)
    assert np.allclose(residual[1:], [-10.4, -18.6, -18.6])
    assert np.allclose(
        jacobian[1:, 1:],
        [[160.0, -40.0, 0.0], [-40.0, 300.0, -60.0], [0.0, -60.0, 360.0]],
    )

    free_fall_params = make_params(
        m=[0.0, 1.0],
        k=[0.0, 0.0],
        b=[0.0, 0.0],
        g=9.8,
        dt=0.01,
    )
    x_step, u_step = implicitEulerStep(
        np.zeros(2), np.zeros(2), free_fall_params
    )
    assert np.allclose(x_step, [0.0, 0.00098])
    assert np.allclose(u_step, [0.0, 0.098])

    x_step, u_step = implicitEulerStep(x_old, u_old, full_params)
    assert np.allclose(x_step, [0.0, 0.096696, 0.116786, 0.126131])
    assert np.allclose(u_step, [0.0, 0.966964, 1.167857, 1.261310])

    one_spring_params = make_params(
        m=[0.0, 1.0],
        k=[0.0, 1.0],
        b=[0.0, 0.0],
        g=0.0,
        dt=0.1,
    )
    x_step, u_step = explicitEulerStep(
        np.array([0.0, 1.0]), np.zeros(2), one_spring_params
    )
    assert np.allclose(x_step, [0.0, 1.0])
    assert np.allclose(u_step, [0.0, -0.1])

    x_step, u_step = explicitEulerStep(
        np.array([0.0, 0.01, 0.03, 0.06]), u_old, full_params
    )
    assert np.allclose(x_step, [0.0, 0.02, 0.05, 0.09])
    assert np.allclose(u_step, [0.0, 1.12, 1.21, 1.24])
    print("Required-function tests: passed")


def validate_question_results():
    """Compare all eleven simulations with the Appendix formulas."""

    _, x1, _ = run_case([1.0], [0.0], [0.0], 1.0e-3, 1.0)
    exact1 = 0.5 * 9.8 * 1.0**2 + 0.5 * 9.8 * 1.0 * 1.0e-3
    check_close("Q01", x1[1, -1], exact1)

    _, x2, _ = run_case(
        [1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3, 1.0e-3, 2.0
    )
    exact2 = 0.5 * 9.8 * 2.0**2 + 0.5 * 9.8 * 2.0 * 1.0e-3
    check_close("Q02", x2[3, -1], exact2)

    _, _, u3 = run_case([1.0], [0.0], [2.0], 0.01, 20.0)
    exact3 = 9.8 / 2.0 * (1.0 - np.exp(-2.0 * 20.0))
    check_close("Q03", u3[1, -1], exact3)

    _, _, u4 = run_case([1.0, 1.0], [0.0, 0.0], [2.0, 2.0], 0.01, 20.0)
    exact4 = terminal_velocity([1.0, 1.0], [2.0, 2.0], node=2)
    check_close("Q04", u4[2, -1], exact4, atol=1.0e-4)

    _, x5, _ = run_case([1.0], [10.0], [1.0], 0.01, 50.0)
    check_close("Q05", x5[1, -1], static_position([1.0], [10.0], 1), 1.0e-4)

    _, x6, _ = run_case([1.0] * 5, [10.0] * 5, [5.0] * 5, 0.01, 100.0)
    check_close(
        "Q06", x6[5, -1], static_position([1.0] * 5, [10.0] * 5, 5), 1.0e-4
    )

    _, x7, _ = run_case(
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
        [5.0, 10.0, 15.0],
        0.01,
        100.0,
    )
    exact7 = static_position([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 3)
    check_close("Q07", x7[3, -1], exact7, atol=1.0e-4)

    _, x8, u8 = run_case([1.0], [1.0], [0.0], 0.1, 100.0)
    omega = 1.0
    equilibrium = 9.8
    w0 = -omega * equilibrium
    w8 = w0 / (1.0 + 1j * omega * 0.1) ** 1000
    exact8 = equilibrium + np.real(w8) / omega
    check_close("Q08", x8[1, -1], exact8)

    _, x9, u9 = run_case([1.0], [1.0], [0.0], 0.01, 100.0)
    energy8 = oscillator_energy(x8[1, -1], u8[1, -1], equilibrium, 1.0, 1.0)
    energy9 = oscillator_energy(x9[1, -1], u9[1, -1], equilibrium, 1.0, 1.0)
    simulated9 = energy9 / energy8
    exact9 = (
        (1.0 + 0.01**2) ** -5000 / (1.0 + 0.1**2) ** -500
    ) ** 2
    check_close("Q09", simulated9, exact9)

    time10, x10, u10 = run_case(
        [1.0], [1.0], [0.0], 0.1, 100.0, method="explicit"
    )
    energy0 = oscillator_energy(0.0, 0.0, equilibrium, 1.0, 1.0)
    energy10 = oscillator_energy(
        x10[1, -1], u10[1, -1], equilibrium, 1.0, 1.0
    )
    check_close("Q10", energy10 / energy0, (1.0 + 0.1**2) ** 1000)

    simulated11 = np.max(x10[1, time10 <= 10.0])
    steps11 = np.arange(101)
    w11 = w0 * (1.0 - 1j * omega * 0.1) ** steps11
    exact11 = np.max(equilibrium + np.real(w11) / omega)
    check_close("Q11", simulated11, exact11)


def main():
    validate_required_functions()
    validate_question_results()
    print("All optional validations passed.")


if __name__ == "__main__":
    main()
