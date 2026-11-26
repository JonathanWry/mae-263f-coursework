"""Coupled hinged-limb, deformable-string, and rigid-arrow simulation.

This is the proposal-level model.  Two rigid limbs rotate about fixed hinges
and store energy in torsional springs.  A massive planar discrete elastic rod
connects the limb tips and interacts with a rigid translating arrow through
unilateral penalty contact.  The reported release is the final contact-opening
event before the first limb reversal.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from archery_prediction import (
    interpolate_at_target,
    required_aim_angle,
    simulate_flight,
)

# Reuse the planar DER bending derivative already verified in Homework 2.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))
from homework_2.elastic_derivatives import bending_gradient_hessian  # noqa: E402


@dataclass(frozen=True)
class CoupledBowParameters:
    """Physical and numerical parameters of the reduced bow model."""

    node_count: int = 15
    launch_dt: float = 1.0e-4
    launch_duration: float = 0.050
    hinge_half_spacing: float = 0.10
    limb_length: float = 0.50
    limb_mass: float = 0.06
    limb_flexural_rigidity: float = 30.0
    limb_reference_angle: float = 1.35
    hinge_damping: float = 0.020
    string_EA: float = 5000.0
    string_mass: float = 0.006
    string_rest_length: float = 1.16
    string_bending_rigidity: float = 1.0e-4
    string_damping: float = 0.020
    arrow_mass: float = 0.025
    arrow_length: float = 0.75
    arrow_diameter: float = 0.006
    contact_stiffness: float = 1.0e7
    draw_length: float = 0.70
    target_distance: float = 70.0
    initial_height: float = 1.50
    target_height: float = 1.50
    release_force_tolerance: float = 1.0e-3
    stop_before_limb_reversal: bool = True

    def __post_init__(self) -> None:
        if self.node_count < 5 or self.node_count % 2 == 0:
            raise ValueError("node_count must be an odd integer of at least 5")
        positive = {
            "launch_dt": self.launch_dt,
            "launch_duration": self.launch_duration,
            "limb_length": self.limb_length,
            "limb_mass": self.limb_mass,
            "limb_flexural_rigidity": self.limb_flexural_rigidity,
            "string_EA": self.string_EA,
            "string_mass": self.string_mass,
            "string_rest_length": self.string_rest_length,
            "string_bending_rigidity": self.string_bending_rigidity,
            "arrow_mass": self.arrow_mass,
            "contact_stiffness": self.contact_stiffness,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def torsional_stiffness(self) -> float:
        """Root-hinge stiffness matched to cantilever tip compliance."""
        return 3.0 * self.limb_flexural_rigidity / self.limb_length

    @property
    def limb_inertia(self) -> float:
        """Uniform rigid-limb moment of inertia about its root hinge."""
        return self.limb_mass * self.limb_length**2 / 3.0

    @property
    def rest_edge_length(self) -> float:
        return self.string_rest_length / (self.node_count - 1)

    @property
    def midpoint(self) -> int:
        return self.node_count // 2

    @property
    def degree_count(self) -> int:
        return 3 + 2 * (self.node_count - 2)

    @property
    def midpoint_x_index(self) -> int:
        return 2 + 2 * (self.midpoint - 1)

    @property
    def midpoint_y_index(self) -> int:
        return self.midpoint_x_index + 1

    @property
    def arrow_x_index(self) -> int:
        return self.degree_count - 1


@dataclass
class StaticEquilibria:
    braced_state: np.ndarray
    full_draw_state: np.ndarray
    brace_x: float
    draw_displacement: np.ndarray
    draw_force: np.ndarray
    braced_residual: float
    full_draw_residual: float


@dataclass
class CoupledArcheryResult:
    parameters: CoupledBowParameters
    time: np.ndarray
    states: list[np.ndarray]
    velocities: np.ndarray
    contact_force: np.ndarray
    arrow_speed: np.ndarray
    limb_energy: np.ndarray
    string_energy: np.ndarray
    contact_energy: np.ndarray
    structure_kinetic_energy: np.ndarray
    arrow_kinetic_energy: np.ndarray
    damping_work: np.ndarray
    energy_balance_error: np.ndarray
    release_time: float
    release_position: float
    release_speed: float
    release_force_before: float
    release_force_after: float
    initial_energy: float
    energy_limit_speed: float
    launch_efficiency: float
    maximum_newton_iterations: int
    equilibria: StaticEquilibria
    required_aim_angle: float
    flight_time: np.ndarray
    flight_position: np.ndarray
    flight_velocity: np.ndarray
    target_time: float
    target_speed: float
    target_height_error: float
    limb_reversal_time: float


def limb_tip(
    theta: float, hinge: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tip position and its first two angle derivatives."""
    direction = np.array([np.cos(theta), np.sin(theta)])
    first = length * np.array([-np.sin(theta), np.cos(theta)])
    second = -length * direction
    return hinge + length * direction, first, second


def hinge_positions(parameters: CoupledBowParameters) -> tuple[np.ndarray, np.ndarray]:
    upper = np.array([0.0, parameters.hinge_half_spacing])
    lower = np.array([0.0, -parameters.hinge_half_spacing])
    return upper, lower


def nodes_and_mapping(
    state: np.ndarray, parameters: CoupledBowParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map generalized coordinates to string nodes and derivatives."""
    upper_hinge, lower_hinge = hinge_positions(parameters)
    upper_tip, upper_first, upper_second = limb_tip(
        state[0], upper_hinge, parameters.limb_length
    )
    lower_tip, lower_first, lower_second = limb_tip(
        state[1], lower_hinge, parameters.limb_length
    )

    nodes = np.empty((parameters.node_count, 2))
    nodes[0] = lower_tip
    nodes[-1] = upper_tip
    node_dof_count = 2 * (parameters.node_count - 2)
    nodes[1:-1] = state[2 : 2 + node_dof_count].reshape(-1, 2)

    mapping = np.zeros((2 * parameters.node_count, parameters.degree_count))
    mapping[0:2, 1] = lower_first
    mapping[-2:, 0] = upper_first
    mapping[2:-2, 2 : 2 + node_dof_count] = np.eye(node_dof_count)
    return nodes, mapping, upper_second, lower_second


def state_from_geometry(
    upper_angle: float,
    lower_angle: float,
    nodes: np.ndarray,
    parameters: CoupledBowParameters,
) -> np.ndarray:
    state = np.empty(parameters.degree_count)
    state[0] = upper_angle
    state[1] = lower_angle
    state[2:-1] = nodes[1:-1].reshape(-1)
    state[parameters.arrow_x_index] = nodes[parameters.midpoint, 0]
    return state


def stretching_edge_energy_derivatives(
    node0: np.ndarray,
    node1: np.ndarray,
    rest_length: float,
    EA: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return bilateral DER stretching energy, derivatives, and axial force."""
    edge = node1 - node0
    length = np.linalg.norm(edge)
    if length <= 0.0:
        raise ValueError("a string edge has zero length")
    extension = length - rest_length
    energy = 0.5 * EA / rest_length * extension**2
    edge_gradient = EA / rest_length * (1.0 - rest_length / length) * edge
    edge_hessian = EA / rest_length * (
        (1.0 - rest_length / length) * np.eye(2)
        + rest_length * np.outer(edge, edge) / length**3
    )
    gradient = np.r_[-edge_gradient, edge_gradient]
    hessian = np.block(
        [[edge_hessian, -edge_hessian], [-edge_hessian, edge_hessian]]
    )
    axial_force = EA * extension / rest_length
    return energy, gradient, hessian, axial_force


def planar_curvature(node0: np.ndarray, node1: np.ndarray, node2: np.ndarray) -> float:
    """Return the planar DER curvature binormal scalar at an internal node."""
    edge0 = node1 - node0
    edge1 = node2 - node1
    tangent0 = edge0 / np.linalg.norm(edge0)
    tangent1 = edge1 / np.linalg.norm(edge1)
    denominator = 1.0 + float(np.dot(tangent0, tangent1))
    if denominator <= 1.0e-12:
        raise ValueError("adjacent string edges are nearly antiparallel")
    return float(
        2.0
        * (tangent0[0] * tangent1[1] - tangent0[1] * tangent1[0])
        / denominator
    )


def coupled_energy_derivatives(
    state: np.ndarray, parameters: CoupledBowParameters
) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    """Assemble limb/string energy, gradient, and Hessian."""
    nodes, mapping, upper_second, lower_second = nodes_and_mapping(
        state, parameters
    )
    node_vector = nodes.reshape(-1)
    node_gradient = np.zeros_like(node_vector)
    node_hessian = np.zeros((len(node_vector), len(node_vector)))
    string_energy = 0.0
    axial_forces = np.zeros(parameters.node_count - 1)
    bending_energy = 0.0

    for edge_index in range(parameters.node_count - 1):
        indices = np.array(
            [
                2 * edge_index,
                2 * edge_index + 1,
                2 * edge_index + 2,
                2 * edge_index + 3,
            ]
        )
        edge_energy, edge_gradient, edge_hessian, axial_force = (
            stretching_edge_energy_derivatives(
                nodes[edge_index],
                nodes[edge_index + 1],
                parameters.rest_edge_length,
                parameters.string_EA,
            )
        )
        string_energy += edge_energy
        node_gradient[indices] += edge_gradient
        node_hessian[np.ix_(indices, indices)] += edge_hessian
        axial_forces[edge_index] = axial_force

    for node_index in range(1, parameters.node_count - 1):
        indices = np.arange(2 * node_index - 2, 2 * node_index + 4)
        local_nodes = node_vector[indices]
        local_gradient, local_hessian = bending_gradient_hessian(
            local_nodes,
            0.0,
            parameters.rest_edge_length,
            parameters.string_bending_rigidity,
        )
        curvature = planar_curvature(
            nodes[node_index - 1], nodes[node_index], nodes[node_index + 1]
        )
        bending_energy += (
            0.5
            * parameters.string_bending_rigidity
            / parameters.rest_edge_length
            * curvature**2
        )
        node_gradient[indices] += local_gradient
        node_hessian[np.ix_(indices, indices)] += local_hessian

    string_energy += bending_energy

    upper_reference = parameters.limb_reference_angle
    lower_reference = -parameters.limb_reference_angle
    angle_difference = np.array(
        [state[0] - upper_reference, state[1] - lower_reference]
    )
    limb_energy = 0.5 * parameters.torsional_stiffness * np.dot(
        angle_difference, angle_difference
    )

    gradient = np.einsum("ia,i->a", mapping, node_gradient)
    gradient[:2] += parameters.torsional_stiffness * angle_difference
    hessian = np.einsum("ia,ij,jb->ab", mapping, node_hessian, mapping)
    hessian[0, 0] += np.dot(node_gradient[-2:], upper_second)
    hessian[1, 1] += np.dot(node_gradient[0:2], lower_second)
    hessian[0, 0] += parameters.torsional_stiffness
    hessian[1, 1] += parameters.torsional_stiffness

    arrow_x = state[parameters.arrow_x_index]
    midpoint_x = nodes[parameters.midpoint, 0]
    contact_gap = arrow_x - midpoint_x
    penetration = max(0.0, -contact_gap)
    contact_force = parameters.contact_stiffness * penetration
    contact_energy = 0.5 * parameters.contact_stiffness * penetration**2
    if penetration > 0.0:
        midpoint_index = parameters.midpoint_x_index
        arrow_index = parameters.arrow_x_index
        gradient[midpoint_index] += contact_force
        gradient[arrow_index] -= contact_force
        hessian[midpoint_index, midpoint_index] += parameters.contact_stiffness
        hessian[arrow_index, arrow_index] += parameters.contact_stiffness
        hessian[midpoint_index, arrow_index] -= parameters.contact_stiffness
        hessian[arrow_index, midpoint_index] -= parameters.contact_stiffness
    hessian = 0.5 * (hessian + hessian.T)

    details: dict[str, np.ndarray | float] = {
        "limb_energy": float(limb_energy),
        "string_energy": float(string_energy),
        "nodes": nodes,
        "axial_forces": axial_forces,
        "tensions": np.maximum(axial_forces, 0.0),
        "string_bending_energy": float(bending_energy),
        "contact_gap": float(contact_gap),
        "contact_force": float(contact_force),
        "contact_energy": float(contact_energy),
    }
    return limb_energy + string_energy + contact_energy, gradient, hessian, details


def mass_and_damping_vectors(
    parameters: CoupledBowParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal generalized mass and damping vectors."""
    edge_mass = parameters.string_mass / (parameters.node_count - 1)
    endpoint_mass = 0.5 * edge_mass
    mass = np.empty(parameters.degree_count)
    mass[0:2] = parameters.limb_inertia + endpoint_mass * parameters.limb_length**2
    mass[2:-1] = edge_mass
    mass[parameters.arrow_x_index] = parameters.arrow_mass

    nodal_damping = parameters.string_damping / (parameters.node_count - 1)
    endpoint_damping = 0.5 * nodal_damping
    damping = np.empty(parameters.degree_count)
    damping[0:2] = (
        parameters.hinge_damping
        + endpoint_damping * parameters.limb_length**2
    )
    damping[2:-1] = nodal_damping
    damping[parameters.arrow_x_index] = 0.0
    return mass, damping


def initial_braced_guess(parameters: CoupledBowParameters) -> np.ndarray:
    upper_angle = parameters.limb_reference_angle
    lower_angle = -parameters.limb_reference_angle
    upper_hinge, lower_hinge = hinge_positions(parameters)
    upper_tip, _, _ = limb_tip(upper_angle, upper_hinge, parameters.limb_length)
    lower_tip, _, _ = limb_tip(lower_angle, lower_hinge, parameters.limb_length)
    fraction = np.linspace(0.0, 1.0, parameters.node_count)[:, None]
    nodes = lower_tip + fraction * (upper_tip - lower_tip)
    return state_from_geometry(upper_angle, lower_angle, nodes, parameters)


def solve_static_equilibrium(
    initial_state: np.ndarray,
    parameters: CoupledBowParameters,
    fixed_values: dict[int, float] | None = None,
    tolerance: float = 1.0e-5,
    maximum_iterations: int = 120,
) -> tuple[np.ndarray, int, float]:
    """Minimize total elastic energy with optional prescribed coordinates."""
    fixed_values = fixed_values or {}
    state = initial_state.copy()
    fixed_indices = np.array(sorted(fixed_values), dtype=int)
    free_indices = np.setdiff1d(np.arange(parameters.degree_count), fixed_indices)

    def impose_fixed(candidate: np.ndarray) -> None:
        for index, value in fixed_values.items():
            candidate[index] = value

    impose_fixed(state)
    for iteration in range(1, maximum_iterations + 1):
        energy, gradient, hessian, _ = coupled_energy_derivatives(state, parameters)
        residual_norm = float(np.linalg.norm(gradient[free_indices], ord=np.inf))
        if residual_norm <= tolerance:
            return state, iteration, residual_norm

        reduced_hessian = hessian[np.ix_(free_indices, free_indices)]
        reduced_gradient = gradient[free_indices]
        regularization = 0.0
        for _ in range(12):
            try:
                direction = np.linalg.solve(
                    reduced_hessian
                    + regularization * np.eye(len(free_indices)),
                    -reduced_gradient,
                )
            except np.linalg.LinAlgError:
                regularization = max(1.0e-8, 10.0 * regularization)
                continue
            if np.dot(reduced_gradient, direction) < 0.0:
                break
            regularization = max(1.0e-8, 10.0 * regularization)
        else:
            raise RuntimeError("could not find a static descent direction")

        diagonal_scale = np.maximum(np.abs(np.diag(reduced_hessian)), 1.0)
        fallback_direction = -reduced_gradient / diagonal_scale
        accepted = False
        for search_direction in (direction, fallback_direction):
            slope = float(np.dot(reduced_gradient, search_direction))
            if slope >= 0.0:
                continue
            step_length = 1.0
            for _ in range(28):
                candidate = state.copy()
                candidate[free_indices] += step_length * search_direction
                impose_fixed(candidate)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    candidate_energy, _, _, _ = coupled_energy_derivatives(
                        candidate, parameters
                    )
                if np.isfinite(candidate_energy) and (
                    candidate_energy <= energy + 1.0e-4 * step_length * slope
                ):
                    state = candidate
                    accepted = True
                    break
                step_length *= 0.5
            if accepted:
                break
        if not accepted:
            raise RuntimeError("static equilibrium line search failed")

    _, gradient, _, _ = coupled_energy_derivatives(state, parameters)
    residual_norm = float(np.linalg.norm(gradient[free_indices], ord=np.inf))
    raise RuntimeError(
        f"static equilibrium did not converge; residual={residual_norm:.3e}"
    )


def solve_equilibria(
    parameters: CoupledBowParameters,
    force_draw_points: int = 13,
) -> StaticEquilibria:
    """Solve braced equilibrium and a continuation to full draw."""
    braced_guess = initial_braced_guess(parameters)
    # Keep the unused arrow ahead of the string during the no-arrow braced solve.
    braced_guess[parameters.arrow_x_index] = 1.0
    braced_state, _, braced_residual = solve_static_equilibrium(
        braced_guess,
        parameters,
        {parameters.arrow_x_index: 1.0},
    )
    braced_nodes, _, _, _ = nodes_and_mapping(braced_state, parameters)
    brace_x = float(braced_nodes[parameters.midpoint, 0])
    braced_state[parameters.arrow_x_index] = brace_x

    draw_displacement = np.linspace(0.0, parameters.draw_length, force_draw_points)
    draw_force = np.zeros_like(draw_displacement)
    state = braced_state.copy()
    residual = braced_residual
    fixed = {
        parameters.midpoint_x_index: brace_x,
        parameters.midpoint_y_index: 0.0,
        parameters.arrow_x_index: brace_x,
    }
    for index, draw in enumerate(draw_displacement):
        fixed[parameters.midpoint_x_index] = brace_x - float(draw)
        fixed[parameters.arrow_x_index] = brace_x - float(draw)
        state[parameters.midpoint_x_index] = fixed[parameters.midpoint_x_index]
        state[parameters.midpoint_y_index] = 0.0
        state[parameters.arrow_x_index] = fixed[parameters.arrow_x_index]
        state, _, residual = solve_static_equilibrium(state, parameters, fixed)
        _, gradient, _, _ = coupled_energy_derivatives(state, parameters)
        draw_force[index] = abs(gradient[parameters.midpoint_x_index])
        # Seed the penalty contact with the static nock force so release starts
        # from the drawn equilibrium instead of generating a first-step impulse.
        state[parameters.arrow_x_index] = (
            fixed[parameters.midpoint_x_index]
            - draw_force[index] / parameters.contact_stiffness
        )

    return StaticEquilibria(
        braced_state=braced_state,
        full_draw_state=state,
        brace_x=brace_x,
        draw_displacement=draw_displacement,
        draw_force=draw_force,
        braced_residual=braced_residual,
        full_draw_residual=residual,
    )


def implicit_dynamic_step(
    state_old: np.ndarray,
    velocity_old: np.ndarray,
    dt: float,
    mass: np.ndarray,
    damping: np.ndarray,
    parameters: CoupledBowParameters,
    fixed_values: dict[int, float] | None = None,
    tolerance: float = 1.0e-7,
    maximum_iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Advance the coupled generalized coordinates by backward Euler."""
    fixed_values = fixed_values or {}
    fixed_indices = np.array(sorted(fixed_values), dtype=int)
    free_indices = np.setdiff1d(np.arange(parameters.degree_count), fixed_indices)
    state = state_old + dt * velocity_old

    def impose_fixed(candidate: np.ndarray) -> None:
        for index, value in fixed_values.items():
            candidate[index] = value

    impose_fixed(state)
    for iteration in range(1, maximum_iterations + 1):
        _, gradient, hessian, _ = coupled_energy_derivatives(state, parameters)
        velocity = (state - state_old) / dt
        residual = (
            mass * (velocity / dt - velocity_old / dt)
            + damping * velocity
            + gradient
        )
        if np.linalg.norm(residual[free_indices], ord=np.inf) <= tolerance:
            return state, velocity, iteration
        jacobian = np.diag(mass / dt**2 + damping / dt) + hessian
        increment = np.linalg.solve(
            jacobian[np.ix_(free_indices, free_indices)],
            residual[free_indices],
        )
        state[free_indices] -= increment
        impose_fixed(state)

    residual_norm = np.linalg.norm(residual[free_indices], ord=np.inf)
    raise RuntimeError(
        f"dynamic Newton solve did not converge; residual={residual_norm:.3e}"
    )


def run_coupled_archery(
    parameters: CoupledBowParameters | None = None,
    *,
    force_draw_points: int = 13,
    compute_flight: bool = True,
    verbose: bool = True,
) -> CoupledArcheryResult:
    """Run the proposal-level launch simulation and optional arrow flight."""
    parameters = parameters or CoupledBowParameters()
    equilibria = solve_equilibria(parameters, force_draw_points)
    base_mass, damping = mass_and_damping_vectors(parameters)
    state = equilibria.full_draw_state.copy()
    velocity = np.zeros(parameters.degree_count)

    step_count = int(round(parameters.launch_duration / parameters.launch_dt))
    time = np.arange(step_count + 1) * parameters.launch_dt
    states = [state.copy()]
    velocities = np.zeros((step_count + 1, parameters.degree_count))
    contact_force = np.zeros(step_count + 1)
    arrow_speed = np.zeros(step_count + 1)
    limb_energy = np.zeros(step_count + 1)
    string_energy = np.zeros(step_count + 1)
    contact_energy = np.zeros(step_count + 1)
    contact_gap = np.zeros(step_count + 1)
    structure_kinetic = np.zeros(step_count + 1)
    arrow_kinetic = np.zeros(step_count + 1)
    damping_work = np.zeros(step_count + 1)
    energy_balance_error = np.zeros(step_count + 1)

    _, _, _, initial_details = coupled_energy_derivatives(state, parameters)
    maximum_newton_iterations = 0
    limb_reversal_time = np.nan
    used_step_count = step_count
    contact_has_occurred = False

    limb_energy[0] = float(initial_details["limb_energy"])
    string_energy[0] = float(initial_details["string_energy"])
    contact_energy[0] = float(initial_details["contact_energy"])
    contact_gap[0] = float(initial_details["contact_gap"])
    contact_force[0] = float(initial_details["contact_force"])
    contact_has_occurred = (
        contact_force[0] > parameters.release_force_tolerance
    )
    initial_energy = limb_energy[0] + string_energy[0] + contact_energy[0]

    for step in range(step_count):
        state_old = state.copy()
        velocity_old = velocity.copy()
        fixed = {parameters.midpoint_y_index: 0.0}

        state, velocity, iterations = implicit_dynamic_step(
            state_old,
            velocity_old,
            parameters.launch_dt,
            base_mass,
            damping,
            parameters,
            fixed,
        )
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)
        velocities[step + 1] = velocity
        states.append(state.copy())

        _, _, _, details = coupled_energy_derivatives(state, parameters)
        contact_force[step + 1] = float(details["contact_force"])
        contact_gap[step + 1] = float(details["contact_gap"])
        arrow_speed[step + 1] = velocity[parameters.arrow_x_index]
        if contact_force[step + 1] > parameters.release_force_tolerance:
            contact_has_occurred = True
        limb_energy[step + 1] = float(details["limb_energy"])
        string_energy[step + 1] = float(details["string_energy"])
        contact_energy[step + 1] = float(details["contact_energy"])
        structure_kinetic[step + 1] = 0.5 * np.sum(
            base_mass[:-1] * velocity[:-1] ** 2
        )
        arrow_kinetic[step + 1] = (
            0.5
            * parameters.arrow_mass
            * velocity[parameters.arrow_x_index] ** 2
        )
        damping_work[step + 1] = damping_work[step] + parameters.launch_dt * np.sum(
            damping * velocity**2
        )
        accounted_energy = (
            limb_energy[step + 1]
            + string_energy[step + 1]
            + contact_energy[step + 1]
            + structure_kinetic[step + 1]
            + arrow_kinetic[step + 1]
            + damping_work[step + 1]
        )
        energy_balance_error[step + 1] = initial_energy - accounted_energy

        if (
            parameters.stop_before_limb_reversal
            and contact_has_occurred
            and velocity_old[0] < 0.0 <= velocity[0]
        ):
            limb_reversal_time = float(time[step + 1])
            # The current state is the first state moving in the reverse
            # direction, so exclude it from the reported launch history.
            used_step_count = step
            break

    kept_count = used_step_count + 1
    time = time[:kept_count]
    states = states[:kept_count]
    velocities = velocities[:kept_count]
    contact_force = contact_force[:kept_count]
    arrow_speed = arrow_speed[:kept_count]
    limb_energy = limb_energy[:kept_count]
    string_energy = string_energy[:kept_count]
    contact_energy = contact_energy[:kept_count]
    contact_gap = contact_gap[:kept_count]
    structure_kinetic = structure_kinetic[:kept_count]
    arrow_kinetic = arrow_kinetic[:kept_count]
    damping_work = damping_work[:kept_count]
    energy_balance_error = energy_balance_error[:kept_count]

    active_contact = contact_force > parameters.release_force_tolerance
    separation_indices = np.flatnonzero(
        active_contact[:-1] & ~active_contact[1:]
    )
    if not len(separation_indices):
        raise RuntimeError("unilateral arrow separation was not detected")
    separation_index = int(separation_indices[-1])
    gap_before = contact_gap[separation_index]
    gap_after = contact_gap[separation_index + 1]
    denominator = gap_after - gap_before
    fraction = (
        -gap_before / denominator if denominator > 0.0 else 1.0
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    release_time = time[separation_index] + fraction * parameters.launch_dt
    release_position = (
        states[separation_index][parameters.arrow_x_index]
        + fraction
        * (
            states[separation_index + 1][parameters.arrow_x_index]
            - states[separation_index][parameters.arrow_x_index]
        )
    )
    release_speed = max(
        0.0,
        velocities[separation_index, parameters.arrow_x_index]
        + fraction
        * (
            velocities[separation_index + 1, parameters.arrow_x_index]
            - velocities[separation_index, parameters.arrow_x_index]
        ),
    )
    release_force_before = contact_force[separation_index]
    release_force_after = contact_force[separation_index + 1]

    energy_limit_speed = np.sqrt(2.0 * initial_energy / parameters.arrow_mass)
    launch_efficiency = (
        0.5 * parameters.arrow_mass * release_speed**2 / initial_energy
    )

    if compute_flight:
        aim_angle = required_aim_angle(
            release_speed,
            parameters.target_distance,
            parameters.initial_height,
            parameters.target_height,
            parameters.arrow_mass,
            parameters.arrow_diameter,
        )
        flight_time, flight_position, flight_velocity = simulate_flight(
            release_speed,
            aim_angle,
            parameters.target_distance,
            parameters.initial_height,
            parameters.arrow_mass,
            parameters.arrow_diameter,
        )
        target_height = float(
            interpolate_at_target(
                flight_position,
                flight_position[:, 1],
                parameters.target_distance,
            )
        )
        target_time = float(
            interpolate_at_target(
                flight_position, flight_time, parameters.target_distance
            )
        )
        target_velocity = interpolate_at_target(
            flight_position, flight_velocity, parameters.target_distance
        )
        target_speed = float(np.linalg.norm(target_velocity))
        target_height_error = target_height - parameters.target_height
    else:
        aim_angle = np.nan
        flight_time = np.empty(0)
        flight_position = np.empty((0, 2))
        flight_velocity = np.empty((0, 2))
        target_time = np.nan
        target_speed = np.nan
        target_height_error = np.nan

    if verbose:
        print("Coupled bow-string-arrow simulation")
        print(
            f"  nodes / edges               : "
            f"{parameters.node_count} / {parameters.node_count - 1}"
        )
        print(f"  beam-equivalent k_theta     : {parameters.torsional_stiffness:.3f} N m/rad")
        print(f"  brace-plane x               : {equilibria.brace_x:.6f} m")
        print(f"  computed full-draw force    : {equilibria.draw_force[-1]:.3f} N")
        print(f"  initial stored energy       : {initial_energy:.3f} J")
        print(f"  force-based release time    : {1e3 * release_time:.3f} ms")
        print(f"  force-based release x       : {release_position:.6f} m")
        print(f"  release speed               : {release_speed:.3f} m/s")
        print(f"  energy-limit speed          : {energy_limit_speed:.3f} m/s")
        print(f"  launch efficiency           : {100.0 * launch_efficiency:.2f}%")
        print(f"  damping work at final time  : {damping_work[-1]:.6f} J")
        print(f"  numerical energy loss       : {energy_balance_error[-1]:.6f} J")
        print(f"  max Newton iterations       : {maximum_newton_iterations}")
        if np.isfinite(limb_reversal_time):
            print(
                f"  stopped before limb reversal: "
                f"{1e3 * limb_reversal_time:.3f} ms"
            )
        if compute_flight:
            print(f"  required 70 m aim angle     : {aim_angle:.3f} deg")

    return CoupledArcheryResult(
        parameters=parameters,
        time=time,
        states=states,
        velocities=velocities,
        contact_force=contact_force,
        arrow_speed=arrow_speed,
        limb_energy=limb_energy,
        string_energy=string_energy,
        contact_energy=contact_energy,
        structure_kinetic_energy=structure_kinetic,
        arrow_kinetic_energy=arrow_kinetic,
        damping_work=damping_work,
        energy_balance_error=energy_balance_error,
        release_time=release_time,
        release_position=release_position,
        release_speed=release_speed,
        release_force_before=float(release_force_before),
        release_force_after=float(release_force_after),
        initial_energy=float(initial_energy),
        energy_limit_speed=float(energy_limit_speed),
        launch_efficiency=float(launch_efficiency),
        maximum_newton_iterations=maximum_newton_iterations,
        equilibria=equilibria,
        required_aim_angle=float(aim_angle),
        flight_time=flight_time,
        flight_position=flight_position,
        flight_velocity=flight_velocity,
        target_time=float(target_time),
        target_speed=float(target_speed),
        target_height_error=float(target_height_error),
        limb_reversal_time=float(limb_reversal_time),
    )


def finite_difference_derivative_check(
    state: np.ndarray,
    parameters: CoupledBowParameters,
    step: float = 1.0e-6,
) -> tuple[float, float]:
    """Compare analytic energy derivatives with centered differences."""
    _, gradient, hessian, _ = coupled_energy_derivatives(state, parameters)
    finite_gradient = np.zeros_like(gradient)
    finite_hessian = np.zeros_like(hessian)
    for index in range(len(state)):
        plus = state.copy()
        minus = state.copy()
        plus[index] += step
        minus[index] -= step
        plus_energy, plus_gradient, _, _ = coupled_energy_derivatives(
            plus, parameters
        )
        minus_energy, minus_gradient, _, _ = coupled_energy_derivatives(
            minus, parameters
        )
        finite_gradient[index] = (plus_energy - minus_energy) / (2.0 * step)
        finite_hessian[:, index] = (plus_gradient - minus_gradient) / (2.0 * step)
    gradient_error = np.linalg.norm(finite_gradient - gradient) / max(
        1.0, np.linalg.norm(finite_gradient)
    )
    hessian_error = np.linalg.norm(finite_hessian - hessian) / max(
        1.0, np.linalg.norm(finite_hessian)
    )
    return float(gradient_error), float(hessian_error)


def beam_theory_tip_check(
    parameters: CoupledBowParameters, tip_force: float = 1.0
) -> tuple[float, float, float]:
    """Compare the hinge surrogate with small-deflection beam theory."""
    beam_deflection = (
        tip_force
        * parameters.limb_length**3
        / (3.0 * parameters.limb_flexural_rigidity)
    )
    angle_change = (
        tip_force * parameters.limb_length / parameters.torsional_stiffness
    )
    reference = parameters.limb_reference_angle
    hinge = np.zeros(2)
    initial_tip, _, _ = limb_tip(reference, hinge, parameters.limb_length)
    rotated_tip, _, _ = limb_tip(
        reference + angle_change, hinge, parameters.limb_length
    )
    normal = np.array([-np.sin(reference), np.cos(reference)])
    hinge_deflection = float(np.dot(rotated_tip - initial_tip, normal))
    relative_error = abs(hinge_deflection - beam_deflection) / beam_deflection
    return beam_deflection, hinge_deflection, float(relative_error)


def run_parameter_sweep(
    base_parameters: CoupledBowParameters | None = None,
) -> list[dict[str, float | str]]:
    """Run simple one-at-a-time sweeps without design constraints."""
    base = base_parameters or CoupledBowParameters()
    cases = {
        "string_EA": [0.5, 1.0, 2.0],
        "string_mass": [0.5, 1.0, 2.0],
        "string_damping": [0.5, 1.0, 2.0],
        "limb_flexural_rigidity": [0.8, 1.0, 1.2],
    }
    rows: list[dict[str, float | str]] = []
    for parameter_name, factors in cases.items():
        baseline_value = float(getattr(base, parameter_name))
        for factor in factors:
            parameters = replace(base, **{parameter_name: baseline_value * factor})
            result = run_coupled_archery(
                parameters,
                force_draw_points=7,
                compute_flight=False,
                verbose=False,
            )
            rows.append(
                {
                    "parameter": parameter_name,
                    "factor": factor,
                    "value": baseline_value * factor,
                    "release_time_ms": 1.0e3 * result.release_time,
                    "release_position_m": result.release_position,
                    "release_speed_m_s": result.release_speed,
                    "efficiency_percent": 100.0 * result.launch_efficiency,
                }
            )
    return rows


def plot_coupled_results(result: CoupledArcheryResult) -> plt.Figure:
    """Plot geometry, forces, energies, force-draw response, and flight."""
    parameters = result.parameters
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    release_index = int(np.searchsorted(result.time, result.release_time))
    snapshots = [0, max(1, release_index // 2), release_index, -1]
    labels = ["full draw", "power stroke", "final release", "bow inertia"]
    upper_hinge, lower_hinge = hinge_positions(parameters)
    axes[0, 0].plot(
        [lower_hinge[0], upper_hinge[0]],
        [lower_hinge[1], upper_hinge[1]],
        color="0.15",
        linewidth=5,
        solid_capstyle="round",
        label="rigid riser",
    )
    for index, label in zip(snapshots, labels):
        state = result.states[index]
        nodes, _, _, _ = nodes_and_mapping(state, parameters)
        upper_tip = nodes[-1]
        lower_tip = nodes[0]
        axes[0, 0].plot(nodes[:, 0], nodes[:, 1], "o-", ms=2, label=label)
        axes[0, 0].plot(
            [lower_hinge[0], lower_tip[0]],
            [lower_hinge[1], lower_tip[1]],
            color="saddlebrown",
        )
        axes[0, 0].plot(
            [upper_hinge[0], upper_tip[0]],
            [upper_hinge[1], upper_tip[1]],
            color="saddlebrown",
        )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("shooting direction x [m]")
    axes[0, 0].set_ylabel("vertical position y [m]")
    axes[0, 0].set_title("Coupled limb-string motion")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(result.time * 1e3, result.arrow_speed, label="arrow speed")
    axes[0, 1].set_xlabel("time [ms]")
    axes[0, 1].set_ylabel("speed [m/s]")
    force_axis = axes[0, 1].twinx()
    force_axis.plot(
        result.time * 1e3,
        result.contact_force,
        color="darkorange",
        label="contact force",
    )
    force_axis.set_ylabel("contact force [N]")
    axes[0, 1].axvline(1e3 * result.release_time, color="black", linestyle="--")
    lines1, labels1 = axes[0, 1].get_legend_handles_labels()
    lines2, labels2 = force_axis.get_legend_handles_labels()
    axes[0, 1].legend(lines1 + lines2, labels1 + labels2)
    axes[0, 1].set_title("Force-based arrow release")

    axes[0, 2].plot(result.time * 1e3, result.limb_energy, label="limb elastic")
    axes[0, 2].plot(result.time * 1e3, result.string_energy, label="string elastic")
    axes[0, 2].plot(
        result.time * 1e3, result.contact_energy, label="contact penalty"
    )
    axes[0, 2].plot(
        result.time * 1e3,
        result.structure_kinetic_energy,
        label="limb/string kinetic",
    )
    axes[0, 2].plot(
        result.time * 1e3, result.arrow_kinetic_energy, label="arrow kinetic"
    )
    axes[0, 2].plot(result.time * 1e3, result.damping_work, label="damping work")
    axes[0, 2].set_xlabel("time [ms]")
    axes[0, 2].set_ylabel("energy [J]")
    axes[0, 2].set_title("Energy partition")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(
        result.equilibria.draw_displacement,
        result.equilibria.draw_force,
        "o-",
        label="static response",
    )
    axes[1, 0].set_xlabel("draw displacement [m]")
    axes[1, 0].set_ylabel("draw force [N]")
    axes[1, 0].set_title("Force-draw curve")
    axes[1, 0].legend()

    angles = np.array([state[:2] for state in result.states])
    axes[1, 1].plot(result.time * 1e3, np.rad2deg(angles[:, 0]), label="upper limb")
    axes[1, 1].plot(result.time * 1e3, np.rad2deg(angles[:, 1]), label="lower limb")
    axes[1, 1].set_xlabel("time [ms]")
    axes[1, 1].set_ylabel("hinge angle [deg]")
    axes[1, 1].set_title("Limb-angle history")
    axes[1, 1].legend()

    if len(result.flight_time):
        axes[1, 2].plot(
            result.flight_position[:, 0],
            result.flight_position[:, 1],
            label="prediction",
        )
        axes[1, 2].plot(
            [parameters.target_distance],
            [parameters.target_height],
            "o",
            color="crimson",
            label="target",
        )
    axes[1, 2].set_xlabel("horizontal distance [m]")
    axes[1, 2].set_ylabel("height [m]")
    axes[1, 2].set_title("Rigid-arrow flight")
    axes[1, 2].legend()
    figure.tight_layout()
    return figure


def animate_coupled_launch(result: CoupledArcheryResult) -> FuncAnimation:
    """Animate the power stroke only through force-based arrow release."""
    parameters = result.parameters
    upper_hinge, lower_hinge = hinge_positions(parameters)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.set_aspect("equal")
    axis.set_xlim(-0.65, 1.65)
    axis.set_ylim(-0.72, 0.72)
    axis.set_xlabel("shooting direction x [m]")
    axis.set_ylabel("vertical position y [m]")
    axis.set_title("Coupled bow-string-arrow launch")
    axis.axvline(
        result.equilibria.brace_x,
        color="0.65",
        linestyle="--",
        linewidth=1.0,
        label="brace plane",
    )
    axis.plot(
        [upper_hinge[0], lower_hinge[0]],
        [upper_hinge[1], lower_hinge[1]],
        color="0.15",
        linewidth=7,
        solid_capstyle="round",
        label="rigid riser",
    )
    axis.plot(
        [upper_hinge[0], lower_hinge[0]],
        [upper_hinge[1], lower_hinge[1]],
        "ko",
        markersize=5,
    )
    upper_limb, = axis.plot([], [], color="saddlebrown", linewidth=4)
    lower_limb, = axis.plot([], [], color="saddlebrown", linewidth=4)
    string_line, = axis.plot([], [], "o-", color="steelblue", markersize=3)
    arrow_line, = axis.plot([], [], color="darkorange", linewidth=3)
    arrow_head, = axis.plot([], [], ">", color="darkorange", markersize=8)
    status = axis.text(
        0.02,
        0.97,
        "",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.legend(loc="lower right")

    frame_count = min(201, len(result.time))
    frame_indices = np.unique(
        np.rint(np.linspace(0, len(result.time) - 1, frame_count)).astype(int)
    )

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        simulation_time = float(result.time[index])
        nodes, _, _, _ = nodes_and_mapping(result.states[index], parameters)
        lower_limb.set_data(
            [lower_hinge[0], nodes[0, 0]],
            [lower_hinge[1], nodes[0, 1]],
        )
        upper_limb.set_data(
            [upper_hinge[0], nodes[-1, 0]],
            [upper_hinge[1], nodes[-1, 1]],
        )
        string_line.set_data(nodes[:, 0], nodes[:, 1])

        arrow_nock_x = float(result.states[index][parameters.arrow_x_index])
        speed = float(result.arrow_speed[index])
        if result.contact_force[index] > parameters.release_force_tolerance:
            phase = "string-arrow contact active"
        elif simulation_time < result.release_time:
            phase = "temporary separation before recontact"
        else:
            phase = "final separation; bow and string inertia continue"
        arrow_tip_x = arrow_nock_x + parameters.arrow_length
        arrow_line.set_data([arrow_nock_x, arrow_tip_x], [0.0, 0.0])
        arrow_head.set_data([arrow_tip_x], [0.0])
        status.set_text(
            f"t = {1e3 * simulation_time:5.2f} ms   "
            f"v = {speed:5.2f} m/s   "
            f"F_contact = {result.contact_force[index]:6.2f} N\n{phase}"
        )
        return upper_limb, lower_limb, string_line, arrow_line, arrow_head, status

    return FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=35,
        repeat=True,
        blit=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=15)
    parser.add_argument("--draw-length", type=float, default=0.70)
    parser.add_argument("--string-EA", type=float, default=5000.0)
    parser.add_argument("--string-mass", type=float, default=0.006)
    parser.add_argument("--limb-EI", type=float, default=30.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    parameters = CoupledBowParameters(
        node_count=args.nodes,
        draw_length=args.draw_length,
        string_EA=args.string_EA,
        string_mass=args.string_mass,
        limb_flexural_rigidity=args.limb_EI,
    )
    result = run_coupled_archery(parameters, verbose=not args.quiet)
    if args.sweep:
        for row in run_parameter_sweep(parameters):
            print(row)
    if not args.no_show:
        if args.animate:
            animation = animate_coupled_launch(result)
            # Keep a live reference until pyplot closes the animation window.
            _ = animation
        else:
            plot_coupled_results(result)
        plt.show()


if __name__ == "__main__":
    main()

