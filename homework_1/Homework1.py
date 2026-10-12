"""MAE 263F Homework 1: modular mass-spring-damper simulator.

Run from the repository root with

    python3 src/Homework1.py

The wall is node 0.  For N masses, all state and parameter arrays have length
N + 1; entry 0 is the fixed wall, and spring/damper i connects nodes i-1 and i.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


# ---------------------------------------------------------------------------
# Two-node element functions
# ---------------------------------------------------------------------------


def gradEs(xa, xb, k):
    """Gradient of E = 0.5*k*(xb-xa)^2 in local order [xa, xb]."""

    return k * (xb - xa) * np.array([-1.0, 1.0])


def hessEs(k):
    """Hessian of one linear spring's elastic energy."""

    return k * np.array([[1.0, -1.0], [-1.0, 1.0]])


def dampingForce(xa_new, xb_new, xa_old, xb_old, b, dt):
    """Two-node damping term evaluated with implicit-Euler velocities."""

    ua_new = (xa_new - xa_old) / dt
    ub_new = (xb_new - xb_old) / dt
    return b * (ub_new - ua_new) * np.array([-1.0, 1.0])


def dampingJacobian(b, dt):
    """Derivative of dampingForce with respect to [xa_new, xb_new]."""

    return (b / dt) * np.array([[1.0, -1.0], [-1.0, 1.0]])


# ---------------------------------------------------------------------------
# Parameter and state validation
# ---------------------------------------------------------------------------


def make_params(m, k, b, dt, g=9.8, tol=1.0e-10, maximum_iter=25):
    """Build the parameter dictionary used by all solver functions."""

    return {
        "m": np.asarray(m, dtype=float),
        "k": np.asarray(k, dtype=float),
        "b": np.asarray(b, dtype=float),
        "g": float(g),
        "dt": float(dt),
        "tol": float(tol),
        "maximum_iter": int(maximum_iter),
    }


def _validate_state(x, u, params, require_positive_mass=False):
    """Check the N+1 array convention used throughout the assignment."""

    m = np.asarray(params["m"], dtype=float)
    k = np.asarray(params["k"], dtype=float)
    b = np.asarray(params["b"], dtype=float)

    if x.ndim != 1 or u.ndim != 1:
        raise ValueError("x and u must be one-dimensional arrays")
    if not (x.size == u.size == m.size == k.size == b.size):
        raise ValueError("x, u, m, k, and b must all have length N + 1")
    if x.size < 2:
        raise ValueError("the chain must contain at least one free node")
    if require_positive_mass and np.any(m[1:] <= 0.0):
        raise ValueError("all free-node masses m[1:] must be positive")
    if params["dt"] <= 0.0:
        raise ValueError("dt must be positive")

    return m, k, b


# ---------------------------------------------------------------------------
# Implicit-Euler residual and Jacobian assembly
# ---------------------------------------------------------------------------


def getFJ(x_new, x_old, u_old, params):
    """Assemble the full implicit-Euler residual f and Jacobian J."""

    x_new = np.asarray(x_new, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    m, k, b = _validate_state(x_new, u_old, params)


    dt = float(params["dt"])
    gravity = float(params.get("g", 9.8))
    N = x_new.size - 1

    # Inertia and gravity. Node 0 is the wall and normally has m[0] = 0.
    u_new = (x_new - x_old) / dt
    f = m * (u_new - u_old) / dt - m * gravity
    J = np.diag(m / dt**2)

    # Element i connects nodes i-1 and i.
    for i in range(1, N + 1):
        idx = np.array([i - 1, i])

        f[idx] += gradEs(x_new[i - 1], x_new[i], k[i])
        J[np.ix_(idx, idx)] += hessEs(k[i])

        f[idx] += dampingForce(
            x_new[i - 1],
            x_new[i],
            x_old[i - 1],
            x_old[i],
            b[i],
            dt,
        )
        J[np.ix_(idx, idx)] += dampingJacobian(b[i], dt)

    return f, J


def implicitEulerStep(x_old, u_old, params):
    """Advance one implicit-Euler step using Newton's method."""

    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    _validate_state(x_old, u_old, params, require_positive_mass=True)

    dt = float(params["dt"])
    tolerance = float(params.get("tol", 1.0e-10))
    maximum_iter = int(params.get("maximum_iter", 25))
    free = np.arange(1, x_old.size)

    x_new = x_old.copy()
    x_new[0] = 0.0

    for _ in range(maximum_iter):
        f, J = getFJ(x_new, x_old, u_old, params)
        delta_x = np.linalg.solve(J[np.ix_(free, free)], f[free])
        x_new[free] -= delta_x

        # Check the residual after updating x_new.
        f_updated, _ = getFJ(x_new, x_old, u_old, params)
        if np.linalg.norm(f_updated[free]) < tolerance:
            u_new = (x_new - x_old) / dt
            x_new[0] = 0.0
            u_new[0] = 0.0
            return x_new, u_new

    raise RuntimeError(
        f"Newton's method did not converge in {maximum_iter} iterations"
    )


# ---------------------------------------------------------------------------
# Explicit Euler and the complete time loop
# ---------------------------------------------------------------------------


def explicitEulerStep(x_old, u_old, params):
    """Advance one forward-Euler step using only old-time values."""

    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    m, k, b = _validate_state(
        x_old, u_old, params, require_positive_mass=True
    )

    dt = float(params["dt"])
    gravity = float(params.get("g", 9.8))
    N = x_old.size - 1

    resistance = np.zeros_like(x_old)
    for i in range(1, N + 1):
        idx = np.array([i - 1, i])
        resistance[idx] += gradEs(x_old[i - 1], x_old[i], k[i])
        resistance[idx] += b[i] * (u_old[i] - u_old[i - 1]) * np.array(
            [-1.0, 1.0]
        )

    acceleration = np.zeros_like(x_old)
    acceleration[1:] = gravity - resistance[1:] / m[1:]

    x_new = x_old + dt * u_old
    u_new = u_old + dt * acceleration
    x_new[0] = 0.0
    u_new[0] = 0.0
    return x_new, u_new


def simulate(x0, u0, params, total_time, method="implicit"):
    """Return time, position history, and velocity history."""

    if total_time < 0.0:
        raise ValueError("total_time must be non-negative")

    step_functions = {
        "implicit": implicitEulerStep,
        "explicit": explicitEulerStep,
    }
    if method not in step_functions:
        raise ValueError("method must be 'implicit' or 'explicit'")

    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    _validate_state(x0, u0, params, require_positive_mass=True)

    dt = float(params["dt"])
    n_steps = round(total_time / dt)
    time = np.arange(n_steps + 1, dtype=float) * dt
    x_history = np.zeros((x0.size, n_steps + 1))
    u_history = np.zeros_like(x_history)
    x_history[:, 0] = x0
    u_history[:, 0] = u0

    step = step_functions[method]
    for n in range(n_steps):
        x_history[:, n + 1], u_history[:, n + 1] = step(
            x_history[:, n], u_history[:, n], params
        )

    return time, x_history, u_history


def main():

    def run_question(m, k, b, dt, total_time, method="implicit"):

        # Prepend the fixed wall entry so every array follows [0, value_1, ..., value_N].
        m_full = np.r_[0.0, np.asarray(m, dtype=float)]
        k_full = np.r_[0.0, np.asarray(k, dtype=float)]
        b_full = np.r_[0.0, np.asarray(b, dtype=float)]
        # The residual has force units.  A 1e-8 N absolute tolerance avoids
        # treating roundoff from the very small dt=1e-3 cases as nonconvergence.
        params = make_params(
            m_full, k_full, b_full, dt=dt, g=9.8, tol=1.0e-8
        )
        x0 = np.zeros(m_full.size)
        u0 = np.zeros(m_full.size)
        return simulate(x0, u0, params, total_time, method=method)

    def report(number, time, x, u, metrics=None):
        """Report the simulated state and any derived numerical quantities."""

        print(f"\nQ{number:02d}")
        print(f"  final time = {time[-1]:.6g} s")
        for node in range(1, x.shape[0]):
            print(
                f"  node {node}: x = {x[node, -1]:.9g} m, "
                f"u = {u[node, -1]:.9g} m/s"
            )
        if metrics is not None:
            for label, value, units in metrics:
                print(f"  {label} = {value:.9g} {units}")

    def oscillator_energy(x, u, equilibrium, mass, stiffness):

        return 0.5 * mass * u**2 + 0.5 * stiffness * (x - equilibrium) ** 2


    # Part A: Free fall -----------------------------------------------------
    time_q1, x_q1, u_q1 = run_question(
        m=[1.0], k=[0.0], b=[0.0], dt=1.0e-3, total_time=1.0
    )
    report(1, time_q1, x_q1, u_q1)

    time_q2, x_q2, u_q2 = run_question(
        m=[1.0, 2.0, 3.0],
        k=[0.0, 0.0, 0.0],
        b=[0.0, 0.0, 0.0],
        dt=1.0e-3,
        total_time=2.0,
    )
    report(2, time_q2, x_q2, u_q2)

    # Part B: Terminal velocity --------------------------------------------
    time_q3, x_q3, u_q3 = run_question(
        m=[1.0], k=[0.0], b=[2.0], dt=0.01, total_time=20.0
    )
    report(3, time_q3, x_q3, u_q3)

    time_q4, x_q4, u_q4 = run_question(
        m=[1.0, 1.0],
        k=[0.0, 0.0],
        b=[2.0, 2.0],
        dt=0.01,
        total_time=20.0,
    )
    report(4, time_q4, x_q4, u_q4)

    # Part C: Static equilibrium -------------------------------------------
    time_q5, x_q5, u_q5 = run_question(
        m=[1.0], k=[10.0], b=[1.0], dt=0.01, total_time=50.0
    )
    report(5, time_q5, x_q5, u_q5)

    time_q6, x_q6, u_q6 = run_question(
        m=[1.0] * 5,
        k=[10.0] * 5,
        b=[5.0] * 5,
        dt=0.01,
        total_time=100.0,
    )
    report(6, time_q6, x_q6, u_q6)

    time_q7, x_q7, u_q7 = run_question(
        m=[1.0, 2.0, 3.0],
        k=[10.0, 20.0, 30.0],
        b=[5.0, 10.0, 15.0],
        dt=0.01,
        total_time=100.0,
    )
    report(7, time_q7, x_q7, u_q7)

    # Part D: Artificial damping in implicit Euler -------------------------
    time_q8, x_q8, u_q8 = run_question(
        m=[1.0], k=[1.0], b=[0.0], dt=0.1, total_time=100.0
    )
    report(8, time_q8, x_q8, u_q8)
    equilibrium = 9.8
    oscillator_mass = 1.0
    oscillator_stiffness = 1.0

    time_q9, x_q9, u_q9 = run_question(
        m=[1.0], k=[1.0], b=[0.0], dt=0.01, total_time=100.0
    )
    energy_dt01 = oscillator_energy(
        x_q8[1, -1],
        u_q8[1, -1],
        equilibrium,
        oscillator_mass,
        oscillator_stiffness,
    )
    energy_dt001 = oscillator_energy(
        x_q9[1, -1],
        u_q9[1, -1],
        equilibrium,
        oscillator_mass,
        oscillator_stiffness,
    )
    energy_ratio = energy_dt001 / energy_dt01
    report(
        9,
        time_q9,
        x_q9,
        u_q9,
        metrics=[
            ("energy at t=100, dt=0.1", energy_dt01, "J"),
            ("energy at t=100, dt=0.01", energy_dt001, "J"),
            ("energy ratio dt=0.01 / dt=0.1", energy_ratio, ""),
        ],
    )

    # Part E: Explicit Euler ------------------------------------------------
    time_q10, x_q10, u_q10 = run_question(
        m=[1.0],
        k=[1.0],
        b=[0.0],
        dt=0.1,
        total_time=100.0,
        method="explicit",
    )
    report(10, time_q10, x_q10, u_q10)

    first_ten_seconds = time_q10 <= 10.0
    x_q11 = x_q10[:, first_ten_seconds]
    u_q11 = u_q10[:, first_ten_seconds]
    time_q11 = time_q10[first_ten_seconds]
    maximum_index = np.argmax(x_q11[1])
    maximum_x_first_ten = x_q11[1, maximum_index]
    report(
        11,
        time_q11,
        x_q11,
        u_q11,
        metrics=[
            ("maximum x1", maximum_x_first_ten, "m"),
            ("time of maximum x1", time_q11[maximum_index], "s"),
            ("u1 at maximum x1", u_q11[1, maximum_index], "m/s"),
        ],
    )

    # Put every comparison belonging to one question on the same axes.
    figure_directory = Path(__file__).resolve().parent / "figures"
    figure_directory.mkdir(exist_ok=True)

    fig9, ax9 = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax9.plot(
        time_q8,
        x_q8[1],
        color="#3366A6",
        linewidth=1.3,
    )
    ax9.plot(
        time_q9,
        x_q9[1],
        color="#D97932",
        linewidth=1.3,
    )
    ax9.axhline(equilibrium, color="#777777", linestyle="--", linewidth=0.9)
    ax9.set(xlabel="Time (s)", ylabel="Position x1 (m)", title="Q9")
    ax9.grid(alpha=0.25)
    ax9.legend(
        handles=[
            Patch(facecolor="#3366A6", edgecolor="none", label="dt = 0.1 s"),
            Patch(facecolor="#D97932", edgecolor="none", label="dt = 0.01 s"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        ncol=2,
    )
    fig9.savefig(figure_directory / "Q09_implicit_timestep_comparison.pdf")
    fig9.savefig(figure_directory / "Q09_implicit_timestep_comparison.svg")
    fig9.savefig(
        figure_directory / "Q09_implicit_timestep_comparison.png", dpi=600
    )

    # Q10 shows the full explicit-Euler run; Q11 enlarges 0--10 s and marks
    # the requested maximum from that same simulation.
    fig10, ax10 = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax10.plot(time_q10, x_q10[1], color="#3366A6", linewidth=1.2)
    ax10.axhline(
        equilibrium,
        color="#777777",
        linestyle="--",
        linewidth=0.9,
    )
    ax10.set(xlabel="Time (s)", ylabel="Position x1 (m)", title="Q10")
    ax10.grid(alpha=0.25)
    fig10.savefig(figure_directory / "Q10_explicit_euler.pdf")
    fig10.savefig(figure_directory / "Q10_explicit_euler.svg")
    fig10.savefig(figure_directory / "Q10_explicit_euler.png", dpi=600)

    fig11, ax11 = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax11.plot(time_q11, x_q11[1], color="#3366A6", linewidth=1.4)
    ax11.scatter(
        time_q11[maximum_index],
        maximum_x_first_ten,
        color="#C44E52",
        s=32,
        zorder=3,
    )
    ax11.axhline(
        equilibrium,
        color="#777777",
        linestyle="--",
        linewidth=0.9,
    )
    ax11.set(
        xlabel="Time (s)",
        ylabel="Position x1 (m)",
        title=f"Q11 (maximum x1 = {maximum_x_first_ten:.3f} m)",
    )
    ax11.grid(alpha=0.25)
    fig11.savefig(figure_directory / "Q11_explicit_euler.pdf")
    fig11.savefig(figure_directory / "Q11_explicit_euler.svg")
    fig11.savefig(figure_directory / "Q11_explicit_euler.png", dpi=600)

    print(f"\nPlots saved in: {figure_directory}")
    plt.show()


if __name__ == "__main__":
    main()
