"""Verification checks for the coupled proposal-level archery model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from coupled_archery import (
    CoupledBowParameters,
    beam_theory_tip_check,
    coupled_energy_derivatives,
    finite_difference_derivative_check,
    run_coupled_archery,
    solve_equilibria,
    stretching_edge_energy_derivatives,
)


def main() -> None:
    parameters = CoupledBowParameters()
    equilibria = solve_equilibria(parameters)

    derivative_state = equilibria.full_draw_state.copy()
    derivative_state[parameters.arrow_x_index] -= 1.0e-4
    gradient_error, hessian_error = finite_difference_derivative_check(
        derivative_state, parameters
    )
    assert gradient_error < 1.0e-7
    assert hessian_error < 1.0e-6

    beam_value, hinge_value, beam_error = beam_theory_tip_check(parameters)
    assert beam_error < 1.0e-4

    compression_energy, compression_gradient, compression_hessian, axial_force = (
        stretching_edge_energy_derivatives(
            np.array([0.0, 0.0]),
            np.array([0.9, 0.0]),
            1.0,
            parameters.string_EA,
        )
    )
    assert compression_energy > 0.0
    assert axial_force < 0.0
    assert np.linalg.norm(compression_gradient) > 0.0
    assert np.all(np.isfinite(compression_hessian))

    result = run_coupled_archery(parameters, verbose=False)
    assert equilibria.braced_residual < 2.0e-5
    assert equilibria.full_draw_residual < 2.0e-5
    assert result.release_force_before > 0.0
    assert result.release_force_after <= 0.0
    gaps = np.array(
        [
            state[parameters.arrow_x_index]
            - coupled_energy_derivatives(state, parameters)[3]["nodes"][
                parameters.midpoint, 0
            ]
            for state in result.states
        ]
    )
    assert np.min(gaps) > -5.0e-5
    active = result.contact_force > parameters.release_force_tolerance
    contact_starts = np.flatnonzero(~active[:-1] & active[1:])
    contact_interval_count = len(contact_starts) + int(active[0])
    assert contact_interval_count >= 2
    length_ratios = []
    for state in result.states:
        nodes = coupled_energy_derivatives(state, parameters)[3]["nodes"]
        length_ratios.append(
            np.sum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))
            / parameters.string_rest_length
        )
    post_release_ratios = np.asarray(length_ratios)[
        result.time >= result.release_time
    ]
    assert np.max(np.abs(post_release_ratios - 1.0)) < 0.03

    stiffer_contact = run_coupled_archery(
        replace(parameters, contact_stiffness=2.0 * parameters.contact_stiffness),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    stiffer_contact_gaps = []
    for state in stiffer_contact.states:
        nodes = coupled_energy_derivatives(state, stiffer_contact.parameters)[3][
            "nodes"
        ]
        stiffer_contact_gaps.append(
            state[stiffer_contact.parameters.arrow_x_index]
            - nodes[stiffer_contact.parameters.midpoint, 0]
        )
    assert abs(stiffer_contact.release_speed - result.release_speed) < 0.02
    assert abs(np.min(stiffer_contact_gaps)) < abs(np.min(gaps))

    stiffer_string_bending = run_coupled_archery(
        replace(
            parameters,
            string_bending_rigidity=10.0 * parameters.string_bending_rigidity,
        ),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    assert abs(stiffer_string_bending.release_speed - result.release_speed) < 0.10
    assert result.release_speed < result.energy_limit_speed
    assert 0.0 < result.launch_efficiency < 1.0
    assert result.energy_balance_error[-1] >= -1.0e-4
    assert result.maximum_newton_iterations < 12
    assert abs(result.target_height_error) < 1.0e-5
    assert np.all(result.equilibria.draw_force >= 0.0)
    assert result.equilibria.draw_force[0] < 1.0e-6
    assert result.equilibria.draw_force[-1] > 0.0

    coarse_mesh = run_coupled_archery(
        replace(parameters, node_count=11),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    fine_mesh = run_coupled_archery(
        replace(parameters, node_count=21),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    assert abs(coarse_mesh.release_speed - fine_mesh.release_speed) < 1.20

    coarse_time = run_coupled_archery(
        replace(parameters, launch_dt=2.0e-4),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    fine_time = run_coupled_archery(
        replace(parameters, launch_dt=5.0e-5),
        force_draw_points=9,
        compute_flight=False,
        verbose=False,
    )
    assert abs(result.release_speed - fine_time.release_speed) < 1.0
    assert coarse_time.energy_balance_error[-1] > result.energy_balance_error[-1]
    assert result.energy_balance_error[-1] > fine_time.energy_balance_error[-1]

    print(
        "Finite-difference derivatives: passed "
        f"(gradient={gradient_error:.3e}, Hessian={hessian_error:.3e})"
    )
    print(
        "Beam-theory limb check: passed "
        f"(beam={beam_value:.9f} m, hinge={hinge_value:.9f} m, "
        f"error={100.0 * beam_error:.5f}%)"
    )
    print("Planar-DER stretching/bending regularization: passed")
    print(
        "Static equilibria: passed "
        f"(braced={equilibria.braced_residual:.3e}, "
        f"full draw={equilibria.full_draw_residual:.3e})"
    )
    print(
        "Force-zero separation: passed "
        f"({result.release_force_before:.4f} -> "
        f"{result.release_force_after:.4f} N)"
    )
    print(
        "Unilateral contact/recontact: passed "
        f"({contact_interval_count} contact intervals, "
        f"minimum gap={1e3 * np.min(gaps):.4f} mm)"
    )
    print(
        "Post-release string length: passed "
        f"(maximum relative change="
        f"{100.0 * np.max(np.abs(post_release_ratios - 1.0)):.3f}%)"
    )
    print(
        "Penalty/bending sensitivity: passed "
        f"(2x contact stiffness: {stiffer_contact.release_speed:.3f} m/s, "
        f"10x string EI: {stiffer_string_bending.release_speed:.3f} m/s)"
    )
    print(
        "Energy accounting: passed "
        f"(damping={result.damping_work[-1]:.6f} J, "
        f"numerical loss={result.energy_balance_error[-1]:.6f} J)"
    )
    print(
        "Mesh convergence: passed "
        f"({coarse_mesh.release_speed:.3f}, {result.release_speed:.3f}, "
        f"{fine_mesh.release_speed:.3f} m/s)"
    )
    print(
        "Time-step convergence: passed "
        f"({coarse_time.release_speed:.3f}, {result.release_speed:.3f}, "
        f"{fine_time.release_speed:.3f} m/s)"
    )
    print(
        "Backward-Euler loss decreases with dt: passed "
        f"({coarse_time.energy_balance_error[-1]:.4f}, "
        f"{result.energy_balance_error[-1]:.4f}, "
        f"{fine_time.energy_balance_error[-1]:.4f} J)"
    )
    print(
        "Default result: "
        f"draw force={result.equilibria.draw_force[-1]:.3f} N, "
        f"release x={result.release_position:.6f} m, "
        f"release speed={result.release_speed:.3f} m/s"
    )


if __name__ == "__main__":
    main()
