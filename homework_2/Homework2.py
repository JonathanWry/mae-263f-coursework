import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np

from elastic_derivatives import (
    bending_gradient_hessian,
    stretching_gradient_hessian,
)


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
# Shared elastic assembly and implicit solver
# ---------------------------------------------------------------------------


def assemble_elastic(q, EI, EA, delta_length):
    """Assemble the gradient and Hessian of all beam elastic energies."""

    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size % 2 != 0:
        raise ValueError("q must be a one-dimensional [x1, y1, ...] vector")

    number_of_nodes = q.size // 2
    gradient = np.zeros_like(q)
    hessian = np.zeros((q.size, q.size))

    for node in range(1, number_of_nodes - 1):
        indices = np.arange(2 * node - 2, 2 * node + 4)
        local_gradient, local_hessian = bending_gradient_hessian(
            q[indices], 0.0, delta_length, EI
        )
        gradient[indices] += local_gradient
        hessian[np.ix_(indices, indices)] += local_hessian

    for edge in range(number_of_nodes - 1):
        indices = np.arange(2 * edge, 2 * edge + 4)
        local_gradient, local_hessian = stretching_gradient_hessian(
            q[indices], delta_length, EA
        )
        gradient[indices] += local_gradient
        hessian[np.ix_(indices, indices)] += local_hessian

    return gradient, hessian


def _residual_jacobian(
    q_new,
    q_old,
    u_old,
    dt,
    mass,
    damping,
    external_force,
    EI,
    EA,
    delta_length,
):
    """Return the full implicit-Euler residual and Jacobian."""

    elastic_gradient, elastic_hessian = assemble_elastic(
        q_new, EI, EA, delta_length
    )
    velocity_new = (q_new - q_old) / dt
    residual = (
        mass * (velocity_new - u_old) / dt
        + damping * velocity_new
        + elastic_gradient
        - external_force
    )
    jacobian = (
        np.diag(mass / dt**2 + damping / dt) + elastic_hessian
    )
    return residual, jacobian


def implicit_euler_step(
    q_old,
    u_old,
    dt,
    mass,
    damping,
    external_force,
    EI,
    EA,
    delta_length,
    free_indices=None,
    tolerance=1.0e-8,
    maximum_iterations=100,
):
    """Advance one step with Newton's method on the requested free DOFs."""

    q_old = np.asarray(q_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    mass = np.asarray(mass, dtype=float)
    damping = np.asarray(damping, dtype=float)
    external_force = np.asarray(external_force, dtype=float)
    if not (
        q_old.shape
        == u_old.shape
        == mass.shape
        == damping.shape
        == external_force.shape
    ):
        raise ValueError("state, mass, damping, and force vectors must match")
    if dt <= 0.0 or np.any(mass <= 0.0):
        raise ValueError("dt and every lumped mass must be positive")

    if free_indices is None:
        free_indices = np.arange(q_old.size)
    else:
        free_indices = np.asarray(free_indices, dtype=int)

    q_new = q_old.copy()
    for iteration in range(1, maximum_iterations + 1):
        residual, jacobian = _residual_jacobian(
            q_new,
            q_old,
            u_old,
            dt,
            mass,
            damping,
            external_force,
            EI,
            EA,
            delta_length,
        )
        free_residual = residual[free_indices]
        if np.linalg.norm(free_residual) <= tolerance:
            velocity_new = (q_new - q_old) / dt
            return q_new, velocity_new, iteration

        free_jacobian = jacobian[np.ix_(free_indices, free_indices)]
        q_new[free_indices] -= np.linalg.solve(
            free_jacobian, free_residual
        )

    raise RuntimeError(
        f"Newton's method did not converge in {maximum_iterations} iterations"
    )


# ---------------------------------------------------------------------------
# Part 1: beam falling in a viscous fluid
# ---------------------------------------------------------------------------


def simulate_falling_beam(
    nv,
    dt,
    total_time,
    visc=1000.0,
    R_all=None,
):
    """Simulate Part 1 and return time, midpoint histories, and final q."""

    if nv < 3 or nv % 2 == 0:
        raise ValueError("nv must be an odd integer of at least 3")
    if total_time < 0.0 or visc < 0.0:
        raise ValueError("total_time and visc must be non-negative")

    number_of_dofs = 2 * nv
    rod_length = 0.10
    delta_length = rod_length / (nv - 1)
    middle_node = (nv - 1) // 2

    radii = np.full(nv, delta_length / 10.0)
    radii[middle_node] = 0.025
    if R_all is not None:
        if R_all <= 0.0:
            raise ValueError("R_all must be positive")
        radii[:] = R_all

    metal_density = 7000.0
    fluid_density = 1000.0
    cross_section_radius = 1.0e-3
    youngs_modulus = 1.0e9
    gravity = 9.8
    EI = youngs_modulus * np.pi * cross_section_radius**4 / 4.0
    EA = youngs_modulus * np.pi * cross_section_radius**2
    tolerance = 1.0e-3 * EI / rod_length**2

    node_mass = 4.0 / 3.0 * np.pi * radii**3 * metal_density
    mass = np.repeat(node_mass, 2)
    damping = np.repeat(6.0 * np.pi * visc * radii, 2)
    external_force = np.zeros(number_of_dofs)
    external_force[1::2] = (
        -4.0
        / 3.0
        * np.pi
        * radii**3
        * (metal_density - fluid_density)
        * gravity
    )

    q = np.zeros(number_of_dofs)
    q[0::2] = np.arange(nv) * delta_length
    velocity = np.zeros(number_of_dofs)
    number_of_steps = round(total_time / dt)
    time = np.arange(number_of_steps + 1, dtype=float) * dt
    midpoint_y = np.zeros(number_of_steps + 1)
    midpoint_velocity = np.zeros(number_of_steps + 1)

    for step in range(1, number_of_steps + 1):
        q, velocity, _ = implicit_euler_step(
            q,
            velocity,
            dt,
            mass,
            damping,
            external_force,
            EI,
            EA,
            delta_length,
            tolerance=tolerance,
        )
        midpoint_y[step] = q[2 * middle_node + 1]
        midpoint_velocity[step] = velocity[2 * middle_node + 1]

    return time, midpoint_y, midpoint_velocity, q


def falling_terminal_velocity(nv, visc=1000.0):
    """Analytical terminal velocity from the force balance in Appendix C."""

    rod_length = 0.10
    middle_radius = 0.025
    small_radius = rod_length / (10.0 * (nv - 1))
    radii = np.full(nv, small_radius)
    radii[(nv - 1) // 2] = middle_radius
    return (
        -2.0
        * (7000.0 - 1000.0)
        * 9.8
        / (9.0 * visc)
        * np.sum(radii**3)
        / np.sum(radii)
    )


# ---------------------------------------------------------------------------
# Part 2: simply supported beam with a point load
# ---------------------------------------------------------------------------


def supported_beam_parameters(nv=50):
    """Return Part 2 geometry, stiffnesses, mass, and index arrays."""

    length = 1.0
    outer_radius = 0.013
    inner_radius = 0.011
    youngs_modulus = 70.0e9
    density = 2700.0
    load_position = 0.75
    delta_length = length / (nv - 1)
    area = np.pi * (outer_radius**2 - inner_radius**2)
    EI = (
        youngs_modulus
        * np.pi
        * (outer_radius**4 - inner_radius**4)
        / 4.0
    )
    EA = youngs_modulus * area
    lumped_mass = area * length * density / (nv - 1)
    loaded_node = int(np.argmin(np.abs(np.arange(nv) * delta_length - load_position)))
    fixed_indices = np.array([0, 1, 2 * nv - 1])
    free_indices = np.setdiff1d(np.arange(2 * nv), fixed_indices)
    return {
        "length": length,
        "load_position": load_position,
        "delta_length": delta_length,
        "EI": EI,
        "EA": EA,
        "lumped_mass": lumped_mass,
        "loaded_node": loaded_node,
        "fixed_indices": fixed_indices,
        "free_indices": free_indices,
    }


def simulate_supported_beam(
    load,
    nv=50,
    dt=1.0e-2,
    total_time=1.0,
):
    """Simulate Part 2 and return time, minimum-y history, and final q."""

    if load < 0.0 or total_time < 0.0:
        raise ValueError("load and total_time must be non-negative")

    parameters = supported_beam_parameters(nv)
    number_of_dofs = 2 * nv
    q = np.zeros(number_of_dofs)
    q[0::2] = np.arange(nv) * parameters["delta_length"]
    velocity = np.zeros(number_of_dofs)
    mass = np.full(number_of_dofs, parameters["lumped_mass"])
    damping = np.zeros(number_of_dofs)
    external_force = np.zeros(number_of_dofs)
    load_dof = 2 * parameters["loaded_node"] + 1
    external_force[load_dof] = -load
    # A tighter force tolerance keeps the low-load end of the Q10 sweep
    # accurate; 1e-3*EI/l^2 is too loose relative to P=20 N.
    tolerance = 1.0e-6 * parameters["EI"] / parameters["length"] ** 2

    number_of_steps = round(total_time / dt)
    time = np.arange(number_of_steps + 1, dtype=float) * dt
    minimum_y = np.zeros(number_of_steps + 1)
    maximum_newton_iterations = 0

    for step in range(1, number_of_steps + 1):
        q, velocity, iterations = implicit_euler_step(
            q,
            velocity,
            dt,
            mass,
            damping,
            external_force,
            parameters["EI"],
            parameters["EA"],
            parameters["delta_length"],
            free_indices=parameters["free_indices"],
            tolerance=tolerance,
        )
        minimum_y[step] = np.min(q[1::2])
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)

    return time, minimum_y, q, maximum_newton_iterations


def euler_bernoulli_deflection(load, d=0.75):
    """Return the positive maximum deflection from the handout's Eq. (6)."""

    parameters = supported_beam_parameters()
    length = parameters["length"]
    c = min(d, length - d)
    return (
        load
        * c
        * (length**2 - c**2) ** 1.5
        / (9.0 * np.sqrt(3.0) * parameters["EI"] * length)
    )


def beam_length(q):
    """Return the sum of all current edge lengths."""

    nodes = np.asarray(q).reshape(-1, 2)
    return np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()


# ---------------------------------------------------------------------------
# Question runner and temporary plots
# ---------------------------------------------------------------------------


def _question_header(number, title):
    """Print a clear boundary before starting one homework question."""

    print("\n" + "=" * 72)
    print(f"Q{number}: {title}")


def main():
    """Run Q1-Q11 in order, reusing simulations without combining output."""

    _question_header(1, "buoyant free fall")
    time_q1, midpoint_y_q1, midpoint_velocity_q1, q_q1 = simulate_falling_beam(
        21, 1.0e-2, 1.0, visc=0.0, R_all=0.025
    )
    print(f"  time [s]                       {time_q1[-1]:.9g}")
    print(f"  midpoint y [m]                 {midpoint_y_q1[-1]:.9g}")
    print(f"  midpoint velocity [m/s]        {midpoint_velocity_q1[-1]:.9g}")
    print(f"  final q size                   {q_q1.size}")

    _question_header(2, "terminal velocity of one sphere")
    time_q2, midpoint_y_q2, midpoint_velocity_q2, q_q2 = simulate_falling_beam(
        21, 1.0e-2, 1.0, visc=1000.0, R_all=0.025
    )
    print(f"  time [s]                       {time_q2[-1]:.9g}")
    print(f"  midpoint y [m]                 {midpoint_y_q2[-1]:.9g}")
    print(f"  midpoint velocity [m/s]        {midpoint_velocity_q2[-1]:.9g}")
    print(f"  final q size                   {q_q2.size}")

    _question_header(3, "terminal velocity of the beam")
    time_q3, midpoint_y_q3, midpoint_velocity_q3, q_q3 = simulate_falling_beam(
        21, 1.0e-2, 50.0
    )
    print(f"  time [s]                       {time_q3[-1]:.9g}")
    print(f"  midpoint terminal velocity     {midpoint_velocity_q3[-1]:.9g} m/s")
    print(f"  final q size                   {q_q3.size}")

    _question_header(4, "position of the beam")
    print("  Reusing the Q3 simulation.")
    print(f"  time [s]                       {time_q3[-1]:.9g}")
    print(f"  midpoint y [m]                 {midpoint_y_q3[-1]:.9g}")

    _question_header(5, "spatial discretization")
    discretization_results = {21: midpoint_velocity_q3[-1]}
    for number_of_nodes in (3, 5, 11):
        _, _, midpoint_velocity, _ = simulate_falling_beam(
            number_of_nodes, 1.0e-2, 50.0
        )
        discretization_results[number_of_nodes] = midpoint_velocity[-1]
    print(
        "  N    deltaL[m]    R_small[m]    "
        "|v_sim|[m/s]    |v_balance|[m/s]"
    )
    for number_of_nodes in (3, 5, 11, 21):
        delta_length = 0.10 / (number_of_nodes - 1)
        analytical_speed = abs(falling_terminal_velocity(number_of_nodes))
        print(
            f"  {number_of_nodes:2d}    {delta_length:10.7f}    "
            f"{delta_length / 10.0:10.7f}    "
            f"{abs(discretization_results[number_of_nodes]):12.9g}    "
            f"{analytical_speed:15.9g}"
        )

    node_counts = np.array([3, 5, 11, 21])
    simulated_speeds = np.array(
        [abs(discretization_results[int(n)]) for n in node_counts]
    )
    balance_speeds = np.array(
        [abs(falling_terminal_velocity(int(n))) for n in node_counts]
    )
    figure5, axes5 = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axes5.plot(node_counts, simulated_speeds, "o-", label="Simulation")
    axes5.plot(node_counts, balance_speeds, "s--", label="Force balance")
    axes5.set_xlabel("Number of nodes, N")
    axes5.set_ylabel("Terminal-speed magnitude [m/s]")
    axes5.set_title("Q05: spatial-discretization comparison")
    axes5.set_xticks(node_counts)
    axes5.legend(frameon=False)
    plt.show()
    plt.close(figure5)

    supported_cache = {}

    def supported(load):
        if load not in supported_cache:
            supported_cache[load] = simulate_supported_beam(load)
        return supported_cache[load]

    _question_header(6, "small load")
    time_q6, minimum_y_q6, q_q6, iterations_q6 = supported(2000.0)
    minimum_node_q6 = int(np.argmin(q_q6[1::2]))
    parameters = supported_beam_parameters()
    loaded_node = parameters["loaded_node"]
    print(f"  final time [s]                  {time_q6[-1]:.9g}")
    print(f"  minimum y [m]                  {minimum_y_q6[-1]:.9g}")
    print(f"  downward-deflection magnitude  {abs(minimum_y_q6[-1]):.9g}")

    _question_header(7, "location of the maximum deflection")
    print("  Reusing the Q6 simulation.")
    print(f"  minimum-y node index           {minimum_node_q6}")
    print(f"  minimum-y node current x [m]   {q_q6[2 * minimum_node_q6]:.9g}")
    print(f"  loaded node index              {loaded_node}")
    print(f"  loaded-node current x [m]      {q_q6[2 * loaded_node]:.9g}")

    nodes_q6 = q_q6.reshape(-1, 2)
    figure67, axes67 = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axes67.axhline(0.0, color="0.75", linestyle="--", label="Initial beam")
    axes67.plot(nodes_q6[:, 0], nodes_q6[:, 1], "o-", markersize=3, label="P=2000 N")
    axes67.plot(
        nodes_q6[loaded_node, 0],
        nodes_q6[loaded_node, 1],
        "v",
        color="#C54E3D",
        markersize=8,
        label="Loaded node",
    )
    axes67.plot(
        nodes_q6[minimum_node_q6, 0],
        nodes_q6[minimum_node_q6, 1],
        "*",
        color="#E09F3E",
        markersize=11,
        label="Minimum-y node",
    )
    axes67.set_xlabel("Current x [m]")
    axes67.set_ylabel("Current y [m]")
    axes67.set_title("Q07: loaded node and minimum-y location")
    axes67.legend(frameon=False)
    plt.show()
    plt.close(figure67)

    _question_header(8, "steady state")
    print("  Reusing the Q6 simulation.")
    print(f"  maximum Newton iterations      {iterations_q6}")
    print(
        "  last five minimum-y values     "
        + ", ".join(f"{value:.9g}" for value in minimum_y_q6[-5:])
    )
    figure8, axes8 = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axes8.plot(time_q6, minimum_y_q6, color="#3366A6", linewidth=1.5)
    axes8.set_xlabel("Time [s]")
    axes8.set_ylabel(r"$y_{max}$ [m]")
    axes8.set_title(r"Q08: $y_{max}(t)$ for $P=2000$ N")
    plt.show()
    plt.close(figure8)

    _question_header(9, "large load")
    _, minimum_y_q9, q_q9, iterations_q9 = supported(20000.0)
    minimum_node_q9 = int(np.argmin(q_q9[1::2]))
    print(f"  minimum y [m]                  {minimum_y_q9[-1]:.9g}")
    print(f"  downward-deflection magnitude  {abs(minimum_y_q9[-1]):.9g}")
    print(f"  minimum-y node index           {minimum_node_q9}")
    print(f"  minimum-y node current x [m]   {q_q9[2 * minimum_node_q9]:.9g}")
    print(f"  maximum Newton iterations      {iterations_q9}")

    _question_header(10, "where beam theory breaks down")
    loads = np.array(
        [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
         2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 20000.0]
    )
    simulated_deflection = np.empty_like(loads)
    for index, load in enumerate(loads):
        simulated_deflection[index] = abs(supported(float(load))[1][-1])
    analytical_deflection = np.array(
        [euler_bernoulli_deflection(load) for load in loads]
    )
    relative_difference_percent = 100.0 * (
        analytical_deflection - simulated_deflection
    ) / analytical_deflection

    print("  P[N]      simulation[m]      EB[m]          e(P) [%]")
    for load, simulation, theory, difference in zip(
        loads,
        simulated_deflection,
        analytical_deflection,
        relative_difference_percent,
    ):
        print(
            f"  {load:7.0f}    {simulation:13.9g}    "
            f"{theory:13.9g}    {difference:10.6f}"
        )

    figure10, axes10 = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axes10.semilogx(
        loads,
        relative_difference_percent,
        "o-",
        color="#3366A6",
        label=r"$e(P)$",
    )
    axes10.axhline(10.0, color="#C54E3D", linestyle="--", label="10% threshold")
    axes10.set_xlabel("Point load, P [N]")
    axes10.set_ylabel(r"Relative difference, $e(P)$ [%]")
    axes10.set_title("Q10: relative difference versus load")
    axes10.legend(frameon=False)
    plt.show()
    plt.close(figure10)

    _question_header(11, "diagnostics for why the predictions differ")
    nodes_q9 = q_q9.reshape(-1, 2)
    final_edge_lengths = np.linalg.norm(np.diff(nodes_q9, axis=0), axis=1)
    spring_extensions = final_edge_lengths - parameters["delta_length"]
    total_extension = np.sum(spring_extensions)
    print("  Reusing the Q9 P=20,000 N simulation.")
    print(f"  last-node x-coordinate [m]     {nodes_q9[-1, 0]:.9g}")
    print(f"  total spring length [m]        {np.sum(final_edge_lengths):.9g}")
    print(f"  total spring-length change [m] {total_extension:.9g}")
    print(f"  relative length change [%]     {100.0 * total_extension / parameters['length']:.9g}")
    print(f"  largest single-spring change   {np.max(np.abs(spring_extensions)):.9g} m")


if __name__ == "__main__":
    main()
