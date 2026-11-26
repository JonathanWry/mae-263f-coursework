"""Verification checks for the distributed planar-DER bow model."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from distributed_bow_archery import (
    DistributedBowParameters,
    _assemble_rod,
    _natural_rod_data,
    distributed_energy_derivatives,
    fixed_clamp_indices,
    initial_state,
    limb_global_indices,
    limb_nodes_from_state,
    run_distributed_bow,
    solve_equilibria,
    tension_only_edge_energy_derivatives,
)


def straight_cantilever_tip_deflection(
    segment_count: int,
    length: float,
    bending_rigidity: float,
    axial_rigidity: float,
    tip_force: float = 0.01,
) -> float:
    """Solve a small-load straight cantilever with the same DER assembly."""
    spacing = length / segment_count
    nodes = np.column_stack(
        (
            np.zeros(segment_count + 2),
            np.arange(-1, segment_count + 1) * spacing,
        )
    )
    state = nodes.reshape(-1)
    global_indices = np.arange(state.size)
    rest, voronoi, natural_curvature = _natural_rod_data(nodes)
    free = np.arange(4, state.size)

    for _ in range(100):
        gradient = np.zeros_like(state)
        hessian = np.zeros((state.size, state.size))
        _assemble_rod(
            state,
            global_indices,
            rest,
            voronoi,
            natural_curvature,
            axial_rigidity,
            bending_rigidity,
            gradient,
            hessian,
        )
        gradient[-2] -= tip_force
        if np.linalg.norm(gradient[free], ord=np.inf) < 5.0e-8:
            break
        state[free] -= np.linalg.solve(
            hessian[np.ix_(free, free)], gradient[free]
        )
    else:
        raise RuntimeError("cantilever verification solve did not converge")
    return float(state[-2])


def directional_derivative_check(
    state: np.ndarray,
    parameters: DistributedBowParameters,
) -> tuple[float, float]:
    """Check assembled derivatives; this verifies code, not physical fidelity."""
    rng = np.random.default_rng(263)
    direction = rng.normal(size=parameters.degree_count)
    direction[fixed_clamp_indices(parameters)] = 0.0
    direction /= np.linalg.norm(direction)
    step = 1.0e-6

    energy, gradient, hessian, _ = distributed_energy_derivatives(
        state, parameters
    )
    energy_plus, gradient_plus, _, _ = distributed_energy_derivatives(
        state + step * direction, parameters
    )
    energy_minus, gradient_minus, _, _ = distributed_energy_derivatives(
        state - step * direction, parameters
    )
    finite_gradient = (energy_plus - energy_minus) / (2.0 * step)
    analytic_gradient = float(np.sum(gradient * direction))
    finite_hessian_action = (gradient_plus - gradient_minus) / (2.0 * step)
    analytic_hessian_action = np.einsum("ij,j->i", hessian, direction)
    gradient_scale = max(abs(finite_gradient), abs(analytic_gradient), 1.0)
    hessian_scale = max(
        np.linalg.norm(finite_hessian_action),
        np.linalg.norm(analytic_hessian_action),
        1.0,
    )
    return (
        abs(finite_gradient - analytic_gradient) / gradient_scale,
        np.linalg.norm(finite_hessian_action - analytic_hessian_action)
        / hessian_scale,
    )


def edge_angles(nodes: np.ndarray) -> np.ndarray:
    edges = np.diff(nodes, axis=0)
    return np.unwrap(np.arctan2(edges[:, 1], edges[:, 0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also run mesh and time-step sensitivity simulations",
    )
    args = parser.parse_args()

    parameters = DistributedBowParameters()
    reference_brace_height = 0.217
    reference_draw_force = 126.9
    reference_draw_energy = 32.16
    reference_arrow_speed = 49.73
    reference_efficiency = 0.702
    natural = initial_state(parameters)
    natural_details = distributed_energy_derivatives(natural, parameters)[3]
    assert float(natural_details["limb_energy"]) < 1.0e-15

    compressed_energy, compressed_gradient, compressed_hessian = (
        tension_only_edge_energy_derivatives(
            np.array([0.0, 0.0, 0.9, 0.0]), 1.0, parameters.string_EA
        )
    )
    stretched_energy, stretched_gradient, _ = (
        tension_only_edge_energy_derivatives(
            np.array([0.0, 0.0, 1.1, 0.0]), 1.0, parameters.string_EA
        )
    )
    assert compressed_energy == 0.0
    assert np.all(compressed_gradient == 0.0)
    assert np.all(compressed_hessian == 0.0)
    assert stretched_energy > 0.0
    assert np.linalg.norm(stretched_gradient) > 0.0

    equilibria = solve_equilibria(parameters)
    derivative_state = equilibria.full_draw_state.copy()
    # Keep the one-sided contact branch active on both finite-difference sides.
    derivative_state[parameters.arrow_x_index] -= 1.0e-4
    gradient_error, hessian_error = directional_derivative_check(
        derivative_state, parameters
    )
    assert gradient_error < 2.0e-7
    assert hessian_error < 2.0e-6

    beam_theory = (
        0.01 * parameters.limb_length**3 / (3.0 * parameters.limb_EI)
    )
    beam_segments = (8, 16, 32)
    beam_values = np.array(
        [
            straight_cantilever_tip_deflection(
                count,
                parameters.limb_length,
                parameters.limb_EI,
                parameters.limb_EA,
            )
            for count in beam_segments
        ]
    )
    beam_errors = np.abs(beam_values / beam_theory - 1.0)
    assert np.all(np.diff(beam_errors) < 0.0)
    assert beam_errors[-1] < 0.05

    result = run_distributed_bow(parameters, verbose=False)
    assert equilibria.braced_residual < 5.0e-5
    assert equilibria.full_draw_residual < 5.0e-5
    assert result.release_force_before > 0.0
    assert result.release_force_after <= parameters.contact_force_tolerance
    assert np.min(result.contact_gap) > -1.0e-4
    assert result.draw_force_constraint_satisfied
    assert result.limb_strain_constraint_satisfied
    assert result.string_strain_constraint_satisfied
    assert 0.0 < result.launch_efficiency < 1.0
    assert result.energy_balance_error[-1] >= -1.0e-4
    assert result.maximum_newton_iterations < 12
    assert np.all(np.diff(equilibria.draw_force) >= -1.0e-5)
    force_draw_work = float(
        np.trapezoid(
            equilibria.draw_force, equilibria.draw_displacement
        )
    )
    force_draw_energy_error = abs(
        force_draw_work - result.available_energy
    ) / result.available_energy
    assert force_draw_energy_error < 0.01
    assert abs(abs(equilibria.brace_x) - reference_brace_height) < 0.02
    assert (
        abs(np.max(equilibria.draw_force) - reference_draw_force)
        / reference_draw_force
        < 0.08
    )
    assert (
        abs(result.available_energy - reference_draw_energy)
        / reference_draw_energy
        < 0.08
    )
    assert (
        abs(result.release_speed - reference_arrow_speed)
        / reference_arrow_speed
        < 0.05
    )
    assert abs(result.launch_efficiency - reference_efficiency) < 0.05

    clamps = fixed_clamp_indices(parameters)
    clamp_reference = result.states[0][clamps]
    clamp_error = max(
        np.max(np.abs(state[clamps] - clamp_reference))
        for state in result.states
    )
    assert clamp_error < 1.0e-14

    natural_upper = limb_nodes_from_state(natural, parameters, upper=True)
    drawn_upper = limb_nodes_from_state(
        equilibria.full_draw_state, parameters, upper=True
    )
    rotation_change = edge_angles(drawn_upper) - edge_angles(natural_upper)
    distributed_rotation_range = float(np.ptp(rotation_change[1:]))
    assert distributed_rotation_range > np.deg2rad(5.0)

    upper_tip_x = np.array(
        [
            limb_nodes_from_state(state, parameters, upper=True)[-1, 0]
            for state in result.states
        ]
    )
    assert np.max(upper_tip_x) < 0.0

    string_bending_energy = np.array(
        [
            float(
                distributed_energy_derivatives(state, parameters)[3][
                    "string_bending_energy"
                ]
            )
            for state in result.states
        ]
    )
    regularization_fraction = float(
        np.max(string_bending_energy) / result.available_energy
    )
    assert regularization_fraction < 1.0e-4

    print(
        "Assembly derivative check: passed "
        f"(directional gradient={gradient_error:.3e}, "
        f"Hessian action={hessian_error:.3e})"
    )
    print("Unstrung reference state: passed (zero limb elastic energy)")
    print(
        "Tension-only cable law: passed "
        "(compressed edge has zero energy and force)"
    )
    print(
        "Euler-Bernoulli cantilever convergence: passed\n"
        + "\n".join(
            f"  {count:2d} segments: delta={value:.9e} m, "
            f"error={100.0 * error:.3f}%"
            for count, value, error in zip(
                beam_segments, beam_values, beam_errors
            )
        )
        + f"\n  theory     : delta={beam_theory:.9e} m"
    )
    print(
        "Static equilibrium: passed "
        f"(braced={equilibria.braced_residual:.3e}, "
        f"full draw={equilibria.full_draw_residual:.3e})"
    )
    print(
        "Unilateral release: passed "
        f"({result.release_force_before:.4f} -> "
        f"{result.release_force_after:.4f} N, "
        f"minimum gap={1e3 * np.min(result.contact_gap):.4f} mm)"
    )
    print(
        "Distributed limb deformation: passed "
        f"(fixed-clamp drift={clamp_error:.1e} m, "
        f"edge-rotation range={np.degrees(distributed_rotation_range):.2f} deg)"
    )
    print(
        "Design constraints: passed "
        f"(draw={np.max(result.equilibria.draw_force):.2f}/"
        f"{parameters.maximum_draw_force:.2f} N, "
        f"limb axial strain={100.0 * np.max(result.maximum_limb_strain):.4f}/"
        f"{100.0 * parameters.maximum_limb_axial_strain:.2f}%, "
        f"string strain={100.0 * np.max(result.maximum_string_strain):.3f}/"
        f"{100.0 * parameters.maximum_string_strain:.1f}%)"
    )
    print(
        "Published Greenhorn-bow reality check: passed\n"
        f"  brace height : {abs(equilibria.brace_x):.4f} vs "
        f"{reference_brace_height:.4f} m\n"
        f"  draw force   : {np.max(equilibria.draw_force):.2f} vs "
        f"{reference_draw_force:.2f} N\n"
        f"  draw energy  : {result.available_energy:.2f} vs "
        f"{reference_draw_energy:.2f} J\n"
        f"  arrow speed  : {result.release_speed:.2f} vs "
        f"{reference_arrow_speed:.2f} m/s\n"
        f"  efficiency   : {100.0 * result.launch_efficiency:.2f} vs "
        f"{100.0 * reference_efficiency:.2f}%"
    )
    print(
        "Force-draw energy consistency: passed "
        f"(integral={force_draw_work:.4f} J, "
        f"state-energy change={result.available_energy:.4f} J, "
        f"error={100.0 * force_draw_energy_error:.3f}%)"
    )
    print(
        "String bending regularization: negligible "
        f"(maximum={np.max(string_bending_energy):.6f} J, "
        f"{100.0 * regularization_fraction:.5f}% of draw energy)"
    )
    print(
        "Post-release limb motion: passed "
        f"(maximum tip x={np.max(upper_tip_x):.4f} m; "
        "the limb tip never crosses the riser plane)"
    )
    print(
        "Default result: "
        f"release speed={result.release_speed:.3f} m/s, "
        f"release x={result.release_position:.5f} m, "
        f"efficiency={100.0 * result.launch_efficiency:.2f}%"
    )

    if args.extended:
        coarse_mesh = run_distributed_bow(
            replace(parameters, limb_segments=8, launch_duration=0.022),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        fine_mesh = run_distributed_bow(
            replace(parameters, limb_segments=12, launch_duration=0.022),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        coarse_time = run_distributed_bow(
            replace(parameters, launch_dt=2.0e-4, launch_duration=0.022),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        fine_time = run_distributed_bow(
            replace(parameters, launch_dt=5.0e-5, launch_duration=0.022),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        stiff_contact = run_distributed_bow(
            replace(
                parameters,
                contact_stiffness=2.0 * parameters.contact_stiffness,
                launch_duration=0.022,
            ),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        weaker_regularization = run_distributed_bow(
            replace(
                parameters,
                string_EI=0.1 * parameters.string_EI,
                launch_duration=0.022,
            ),
            force_draw_points=9,
            compute_flight=False,
            verbose=False,
        )
        assert abs(coarse_mesh.release_speed - result.release_speed) < 5.0
        assert abs(fine_mesh.release_speed - result.release_speed) < 5.0
        assert abs(fine_time.release_speed - result.release_speed) < 3.0
        assert abs(stiff_contact.release_speed - result.release_speed) < 0.10
        assert np.min(stiff_contact.contact_gap) > np.min(result.contact_gap)
        assert (
            abs(weaker_regularization.release_speed - result.release_speed)
            < 0.10
        )

        def loss_at_release(simulation) -> float:
            index = int(
                np.argmin(
                    np.abs(simulation.time - simulation.release_time)
                )
            )
            return float(simulation.energy_balance_error[index])

        coarse_loss = loss_at_release(coarse_time)
        default_loss = loss_at_release(result)
        fine_loss = loss_at_release(fine_time)
        assert coarse_loss > default_loss > fine_loss
        print(
            "Mesh sensitivity (8/10/12 limb segments): "
            f"{coarse_mesh.release_speed:.3f}, {result.release_speed:.3f}, "
            f"{fine_mesh.release_speed:.3f} m/s"
        )
        print(
            "Time-step sensitivity (0.10/0.05 ms): "
            f"{result.release_speed:.3f}, {fine_time.release_speed:.3f} m/s"
        )
        print(
            "Backward-Euler release loss (0.20/0.10/0.05 ms): "
            f"{coarse_loss:.4f}, {default_loss:.4f}, {fine_loss:.4f} J"
        )
        print(
            "Contact-stiffness sensitivity: passed "
            f"(1x/2x speed={result.release_speed:.3f}/"
            f"{stiff_contact.release_speed:.3f} m/s, "
            f"penetration={-1e3 * np.min(result.contact_gap):.4f}/"
            f"{-1e3 * np.min(stiff_contact.contact_gap):.4f} mm)"
        )
        print(
            "String-EI regularization sensitivity: passed "
            f"(1x/0.1x speed={result.release_speed:.3f}/"
            f"{weaker_regularization.release_speed:.3f} m/s)"
        )


if __name__ == "__main__":
    main()
