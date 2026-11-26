"""Distributed planar-DER bow, tension-only string, and rigid-arrow model.

The rigid riser is fixed.  Each limb is a planar discrete elastic rod with
distributed mass, axial stiffness, bending stiffness, and a prescribed
unstrung reference shape.  A tension-only discrete string connects the tips and
interacts with an independent rigid arrow through unilateral penalty contact.
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))
from homework_2.elastic_derivatives import (  # noqa: E402
    bending_gradient_hessian,
    stretching_gradient_hessian,
)


@dataclass(frozen=True)
class DistributedBowParameters:
    """Physical and numerical parameters for the distributed-bow model."""

    # The dimensional defaults follow the Greenhorn TD 350 case in Tuijn and
    # Kooi (European Journal of Physics 13, 127--134, 1992): 68 in bow,
    # 42.9 cm effective draw, 18.1 g arrow, 6.55 g Dacron string, and
    # 260 N per percent string elongation.  limb_EI and string_rest_length are
    # effective DER values calibrated to its static force, energy, and brace
    # height; they are not claimed as measured material properties.
    limb_segments: int = 10
    string_node_count: int = 21
    launch_dt: float = 1.0e-4
    launch_duration: float = 0.040
    riser_half_length: float = 0.10
    limb_length: float = 0.765
    limb_natural_turn: float = 0.0
    limb_mass: float = 0.093
    limb_EA: float = 2.0e6
    limb_EI: float = 68.0
    limb_damping: float = 0.080
    string_rest_length: float = 1.645
    string_mass: float = 0.00655
    string_EA: float = 26000.0
    string_EI: float = 1.0e-4
    string_damping: float = 0.020
    arrow_mass: float = 0.0181
    arrow_length: float = 0.75
    arrow_diameter: float = 0.006
    contact_stiffness: float = 1.0e7
    contact_force_tolerance: float = 1.0e-3
    bracing_pull: float = 0.70
    draw_length: float = 0.429
    maximum_draw_force: float = 160.0
    maximum_limb_axial_strain: float = 5.0e-3
    maximum_string_strain: float = 0.04
    target_distance: float = 70.0
    initial_height: float = 1.50
    target_height: float = 1.50

    def __post_init__(self) -> None:
        if self.limb_segments < 4:
            raise ValueError("limb_segments must be at least 4")
        if self.string_node_count < 5 or self.string_node_count % 2 == 0:
            raise ValueError("string_node_count must be odd and at least 5")
        positive = {
            "launch_dt": self.launch_dt,
            "launch_duration": self.launch_duration,
            "limb_length": self.limb_length,
            "limb_mass": self.limb_mass,
            "limb_EA": self.limb_EA,
            "limb_EI": self.limb_EI,
            "string_rest_length": self.string_rest_length,
            "string_mass": self.string_mass,
            "string_EA": self.string_EA,
            "string_EI": self.string_EI,
            "arrow_mass": self.arrow_mass,
            "contact_stiffness": self.contact_stiffness,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def limb_node_count(self) -> int:
        # One fixed ghost node, one fixed root node, then limb_segments nodes.
        return self.limb_segments + 2

    @property
    def upper_start(self) -> int:
        return 0

    @property
    def lower_start(self) -> int:
        return 2 * self.limb_node_count

    @property
    def string_interior_start(self) -> int:
        return 4 * self.limb_node_count

    @property
    def arrow_x_index(self) -> int:
        return self.degree_count - 1

    @property
    def degree_count(self) -> int:
        return 4 * self.limb_node_count + 2 * (self.string_node_count - 2) + 1

    @property
    def string_midpoint(self) -> int:
        return self.string_node_count // 2

    @property
    def string_rest_edge_length(self) -> float:
        return self.string_rest_length / (self.string_node_count - 1)


@dataclass
class DistributedStaticEquilibria:
    braced_state: np.ndarray
    full_draw_state: np.ndarray
    brace_x: float
    draw_displacement: np.ndarray
    draw_force: np.ndarray
    braced_residual: float
    full_draw_residual: float


@dataclass
class DistributedBowResult:
    parameters: DistributedBowParameters
    time: np.ndarray
    states: list[np.ndarray]
    velocities: np.ndarray
    contact_force: np.ndarray
    contact_gap: np.ndarray
    arrow_speed: np.ndarray
    maximum_limb_strain: np.ndarray
    maximum_string_strain: np.ndarray
    limb_energy: np.ndarray
    string_energy: np.ndarray
    contact_energy: np.ndarray
    kinetic_energy: np.ndarray
    arrow_kinetic_energy: np.ndarray
    damping_work: np.ndarray
    energy_balance_error: np.ndarray
    release_time: float
    release_position: float
    release_speed: float
    release_force_before: float
    release_force_after: float
    initial_energy: float
    braced_energy: float
    available_energy: float
    launch_efficiency: float
    draw_force_constraint_satisfied: bool
    limb_strain_constraint_satisfied: bool
    string_strain_constraint_satisfied: bool
    maximum_newton_iterations: int
    equilibria: DistributedStaticEquilibria
    required_aim_angle: float
    flight_time: np.ndarray
    flight_position: np.ndarray
    flight_velocity: np.ndarray
    target_time: float
    target_speed: float
    target_height_error: float


def _curvature(nodes: np.ndarray, index: int) -> float:
    edge0 = nodes[index] - nodes[index - 1]
    edge1 = nodes[index + 1] - nodes[index]
    tangent0 = edge0 / np.linalg.norm(edge0)
    tangent1 = edge1 / np.linalg.norm(edge1)
    denominator = 1.0 + float(np.dot(tangent0, tangent1))
    if denominator <= 1.0e-12:
        raise ValueError("adjacent rod edges are nearly antiparallel")
    return float(
        2.0
        * (tangent0[0] * tangent1[1] - tangent0[1] * tangent1[0])
        / denominator
    )


def natural_limb_nodes(
    parameters: DistributedBowParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper and lower unstrung centerlines, including clamp ghosts."""
    if abs(parameters.limb_natural_turn) < 1.0e-10:
        spacing = parameters.limb_length / parameters.limb_segments
        arc_coordinates = (
            np.arange(-1, parameters.limb_segments + 1) * spacing
        )
        upper = np.column_stack(
            (
                np.zeros_like(arc_coordinates),
                parameters.riser_half_length + arc_coordinates,
            )
        )
        lower = upper.copy()
        lower[:, 1] *= -1.0
        return upper, lower
    segment_angle = parameters.limb_natural_turn / parameters.limb_segments
    radius = parameters.limb_length / parameters.limb_natural_turn
    angles = np.arange(-1, parameters.limb_segments + 1) * segment_angle
    upper = np.column_stack(
        (
            radius * (1.0 - np.cos(angles)),
            parameters.riser_half_length + radius * np.sin(angles),
        )
    )
    lower = upper.copy()
    lower[:, 1] *= -1.0
    return upper, lower


def limb_global_indices(
    parameters: DistributedBowParameters, *, upper: bool
) -> np.ndarray:
    start = parameters.upper_start if upper else parameters.lower_start
    return np.arange(start, start + 2 * parameters.limb_node_count)


def limb_nodes_from_state(
    state: np.ndarray,
    parameters: DistributedBowParameters,
    *,
    upper: bool,
) -> np.ndarray:
    indices = limb_global_indices(parameters, upper=upper)
    return state[indices].reshape(-1, 2)


def string_dof_indices(parameters: DistributedBowParameters) -> np.ndarray:
    """Return global x/y indices for every string node."""
    indices = np.empty((parameters.string_node_count, 2), dtype=int)
    limb_count = parameters.limb_node_count
    lower_tip_start = parameters.lower_start + 2 * (limb_count - 1)
    upper_tip_start = parameters.upper_start + 2 * (limb_count - 1)
    indices[0] = [lower_tip_start, lower_tip_start + 1]
    indices[-1] = [upper_tip_start, upper_tip_start + 1]
    interior = np.arange(
        parameters.string_interior_start,
        parameters.string_interior_start + 2 * (parameters.string_node_count - 2),
    )
    indices[1:-1] = interior.reshape(-1, 2)
    return indices


def string_nodes_from_state(
    state: np.ndarray, parameters: DistributedBowParameters
) -> np.ndarray:
    return state[string_dof_indices(parameters)]


def fixed_clamp_indices(parameters: DistributedBowParameters) -> np.ndarray:
    upper = limb_global_indices(parameters, upper=True)[:4]
    lower = limb_global_indices(parameters, upper=False)[:4]
    return np.r_[upper, lower]


def initial_state(parameters: DistributedBowParameters) -> np.ndarray:
    upper, lower = natural_limb_nodes(parameters)
    state = np.zeros(parameters.degree_count)
    state[limb_global_indices(parameters, upper=True)] = upper.reshape(-1)
    state[limb_global_indices(parameters, upper=False)] = lower.reshape(-1)
    fraction = np.linspace(0.0, 1.0, parameters.string_node_count)[:, None]
    string_nodes = lower[-1] + fraction * (upper[-1] - lower[-1])
    interior_indices = string_dof_indices(parameters)[1:-1].reshape(-1)
    state[interior_indices] = string_nodes[1:-1].reshape(-1)
    state[parameters.arrow_x_index] = string_nodes[parameters.string_midpoint, 0]
    return state


def _natural_rod_data(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rest_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    voronoi_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
    natural_curvatures = np.array(
        [_curvature(nodes, index) for index in range(1, len(nodes) - 1)]
    )
    return rest_lengths, voronoi_lengths, natural_curvatures


def tension_only_edge_energy_derivatives(
    q_local: np.ndarray,
    rest_length: float,
    axial_rigidity: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return one cable-edge energy and derivatives with no compression."""
    edge = q_local[2:4] - q_local[0:2]
    length = float(np.linalg.norm(edge))
    if length <= 0.0:
        raise ValueError("a string edge has zero length")
    if length <= rest_length:
        return 0.0, np.zeros(4), np.zeros((4, 4))
    strain = (length - rest_length) / rest_length
    energy = 0.5 * axial_rigidity * rest_length * strain**2
    gradient, hessian = stretching_gradient_hessian(
        q_local, rest_length, axial_rigidity
    )
    return float(energy), gradient, hessian


def _assemble_rod(
    state: np.ndarray,
    global_indices: np.ndarray,
    rest_lengths: np.ndarray,
    voronoi_lengths: np.ndarray,
    natural_curvatures: np.ndarray,
    EA: float,
    EI: float,
    gradient: np.ndarray,
    hessian: np.ndarray,
    tension_only: bool = False,
) -> tuple[float, float, float]:
    nodes = state[global_indices].reshape(-1, 2)
    stretching_energy = 0.0
    bending_energy = 0.0
    maximum_strain = 0.0

    for edge_index, rest_length in enumerate(rest_lengths):
        local_global = global_indices[2 * edge_index : 2 * edge_index + 4]
        local = state[local_global]
        length = np.linalg.norm(local[2:4] - local[0:2])
        strain = (length - rest_length) / rest_length
        if tension_only:
            maximum_strain = max(maximum_strain, float(max(0.0, strain)))
        else:
            maximum_strain = max(maximum_strain, abs(float(strain)))
        if tension_only:
            edge_energy, local_gradient, local_hessian = (
                tension_only_edge_energy_derivatives(local, rest_length, EA)
            )
            stretching_energy += edge_energy
        else:
            stretching_energy += 0.5 * EA * rest_length * strain**2
            local_gradient, local_hessian = stretching_gradient_hessian(
                local, rest_length, EA
            )
        gradient[local_global] += local_gradient
        hessian[np.ix_(local_global, local_global)] += local_hessian

    for local_index in range(1, len(nodes) - 1):
        local_global = global_indices[2 * local_index - 2 : 2 * local_index + 4]
        curvature = _curvature(nodes, local_index)
        curvature_difference = curvature - natural_curvatures[local_index - 1]
        voronoi = voronoi_lengths[local_index - 1]
        bending_energy += 0.5 * EI / voronoi * curvature_difference**2
        local_gradient, local_hessian = bending_gradient_hessian(
            state[local_global],
            natural_curvatures[local_index - 1],
            voronoi,
            EI,
        )
        gradient[local_global] += local_gradient
        hessian[np.ix_(local_global, local_global)] += local_hessian

    return stretching_energy + bending_energy, bending_energy, maximum_strain


def distributed_energy_derivatives(
    state: np.ndarray, parameters: DistributedBowParameters
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float | np.ndarray]]:
    """Assemble limb, string, and unilateral-contact energies."""
    gradient = np.zeros(parameters.degree_count)
    hessian = np.zeros((parameters.degree_count, parameters.degree_count))
    natural_upper, natural_lower = natural_limb_nodes(parameters)
    upper_data = _natural_rod_data(natural_upper)
    lower_data = _natural_rod_data(natural_lower)

    upper_energy, upper_bending, upper_strain = _assemble_rod(
        state,
        limb_global_indices(parameters, upper=True),
        *upper_data,
        parameters.limb_EA,
        parameters.limb_EI,
        gradient,
        hessian,
    )
    lower_energy, lower_bending, lower_strain = _assemble_rod(
        state,
        limb_global_indices(parameters, upper=False),
        *lower_data,
        parameters.limb_EA,
        parameters.limb_EI,
        gradient,
        hessian,
    )

    string_indices = string_dof_indices(parameters).reshape(-1)
    string_nodes = state[string_indices].reshape(-1, 2)
    string_rest = np.full(
        parameters.string_node_count - 1, parameters.string_rest_edge_length
    )
    string_voronoi = np.full(
        parameters.string_node_count - 2, parameters.string_rest_edge_length
    )
    string_curvature = np.zeros(parameters.string_node_count - 2)
    string_energy, string_bending, string_strain = _assemble_rod(
        state,
        string_indices,
        string_rest,
        string_voronoi,
        string_curvature,
        parameters.string_EA,
        parameters.string_EI,
        gradient,
        hessian,
        tension_only=True,
    )

    midpoint_x_index = string_dof_indices(parameters)[parameters.string_midpoint, 0]
    arrow_x = state[parameters.arrow_x_index]
    midpoint_x = state[midpoint_x_index]
    contact_gap = arrow_x - midpoint_x
    penetration = max(0.0, -contact_gap)
    contact_force = parameters.contact_stiffness * penetration
    contact_energy = 0.5 * parameters.contact_stiffness * penetration**2
    if penetration > 0.0:
        arrow_index = parameters.arrow_x_index
        gradient[midpoint_x_index] += contact_force
        gradient[arrow_index] -= contact_force
        hessian[midpoint_x_index, midpoint_x_index] += parameters.contact_stiffness
        hessian[arrow_index, arrow_index] += parameters.contact_stiffness
        hessian[midpoint_x_index, arrow_index] -= parameters.contact_stiffness
        hessian[arrow_index, midpoint_x_index] -= parameters.contact_stiffness

    hessian = 0.5 * (hessian + hessian.T)
    limb_energy = upper_energy + lower_energy
    details: dict[str, float | np.ndarray] = {
        "upper_nodes": limb_nodes_from_state(state, parameters, upper=True),
        "lower_nodes": limb_nodes_from_state(state, parameters, upper=False),
        "string_nodes": string_nodes,
        "limb_energy": float(limb_energy),
        "limb_bending_energy": float(upper_bending + lower_bending),
        "string_energy": float(string_energy),
        "string_bending_energy": float(string_bending),
        "contact_energy": float(contact_energy),
        "contact_force": float(contact_force),
        "contact_gap": float(contact_gap),
        "maximum_limb_strain": max(upper_strain, lower_strain),
        "maximum_string_strain": float(string_strain),
    }
    return limb_energy + string_energy + contact_energy, gradient, hessian, details


def mass_and_damping(
    parameters: DistributedBowParameters,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.zeros(parameters.degree_count)
    damping = np.zeros(parameters.degree_count)

    limb_edge_mass = parameters.limb_mass / parameters.limb_segments
    for upper in (True, False):
        node_indices = limb_global_indices(parameters, upper=upper).reshape(-1, 2)
        nodal_mass = np.zeros(parameters.limb_node_count)
        # Physical limb begins at node 1; ghost node 0 has no mass.
        nodal_mass[1] += 0.5 * limb_edge_mass
        nodal_mass[-1] += 0.5 * limb_edge_mass
        nodal_mass[2:-1] += limb_edge_mass
        for node_index, dofs in enumerate(node_indices):
            mass[dofs] += nodal_mass[node_index]
            damping[dofs] += parameters.limb_damping / parameters.limb_segments

    string_edge_mass = parameters.string_mass / (parameters.string_node_count - 1)
    string_mass = np.full(parameters.string_node_count, string_edge_mass)
    string_mass[[0, -1]] *= 0.5
    for node_index, dofs in enumerate(string_dof_indices(parameters)):
        mass[dofs] += string_mass[node_index]
        damping[dofs] += parameters.string_damping / (
            parameters.string_node_count - 1
        )

    mass[parameters.arrow_x_index] = parameters.arrow_mass
    damping[parameters.arrow_x_index] = 0.0
    return mass, damping


def solve_static_equilibrium(
    initial: np.ndarray,
    parameters: DistributedBowParameters,
    fixed_values: dict[int, float],
    tolerance: float = 5.0e-5,
    maximum_iterations: int = 160,
) -> tuple[np.ndarray, int, float]:
    state = initial.copy()
    fixed_indices = np.array(sorted(fixed_values), dtype=int)
    free_indices = np.setdiff1d(np.arange(parameters.degree_count), fixed_indices)

    def impose_fixed(candidate: np.ndarray) -> None:
        for index, value in fixed_values.items():
            candidate[index] = value

    impose_fixed(state)
    for iteration in range(1, maximum_iterations + 1):
        energy, gradient, hessian, _ = distributed_energy_derivatives(
            state, parameters
        )
        residual = float(np.linalg.norm(gradient[free_indices], ord=np.inf))
        if residual <= tolerance:
            return state, iteration, residual

        reduced_gradient = gradient[free_indices]
        reduced_hessian = hessian[np.ix_(free_indices, free_indices)]
        regularization = 0.0
        for _ in range(14):
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

        accepted = False
        slope = float(np.dot(reduced_gradient, direction))
        step_length = 1.0
        for _ in range(32):
            candidate = state.copy()
            candidate[free_indices] += step_length * direction
            impose_fixed(candidate)
            try:
                candidate_energy = distributed_energy_derivatives(
                    candidate, parameters
                )[0]
            except ValueError:
                candidate_energy = np.inf
            if np.isfinite(candidate_energy) and (
                candidate_energy <= energy + 1.0e-4 * step_length * slope
            ):
                state = candidate
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            diagonal = np.maximum(np.abs(np.diag(reduced_hessian)), 1.0)
            state[free_indices] -= 1.0e-2 * reduced_gradient / diagonal
            impose_fixed(state)

    residual = float(
        np.linalg.norm(
            distributed_energy_derivatives(state, parameters)[1][free_indices],
            ord=np.inf,
        )
    )
    raise RuntimeError(f"static solve did not converge; residual={residual:.3e}")


def solve_equilibria(
    parameters: DistributedBowParameters,
    force_draw_points: int = 17,
) -> DistributedStaticEquilibria:
    state = initial_state(parameters)
    clamps = fixed_clamp_indices(parameters)
    fixed = {int(index): float(state[index]) for index in clamps}
    state[parameters.arrow_x_index] = 1.0
    fixed[parameters.arrow_x_index] = 1.0
    relaxed_state, _, _ = solve_static_equilibrium(
        state, parameters, fixed
    )
    string_indices = string_dof_indices(parameters)
    midpoint_x_index, midpoint_y_index = string_indices[parameters.string_midpoint]
    relaxed_x = float(relaxed_state[midpoint_x_index])

    # Stringing is a path-dependent operation.  Starting directly from the
    # unstrung natural shape finds the wrong, forward equilibrium branch.  Pull
    # the string through the bow first, then remove the midpoint-x constraint
    # to obtain the metastable physical braced branch.
    state = relaxed_state.copy()
    fixed = {int(index): float(state[index]) for index in clamps}
    fixed[midpoint_y_index] = 0.0
    fixed[parameters.arrow_x_index] = 1.0
    for stringing_displacement in np.linspace(
        0.0, parameters.bracing_pull, force_draw_points
    ):
        prescribed_x = relaxed_x - float(stringing_displacement)
        fixed[midpoint_x_index] = prescribed_x
        state[midpoint_x_index] = prescribed_x
        state[midpoint_y_index] = 0.0
        state[parameters.arrow_x_index] = 1.0
        state, _, _ = solve_static_equilibrium(state, parameters, fixed)

    fixed.pop(midpoint_x_index)
    braced_state, _, braced_residual = solve_static_equilibrium(
        state, parameters, fixed
    )
    brace_x = float(braced_state[midpoint_x_index])
    if brace_x >= 0.0:
        raise RuntimeError(
            "stringing path returned to the forward equilibrium branch; "
            "adjust string_rest_length or bracing_pull"
        )
    braced_state[parameters.arrow_x_index] = brace_x

    draw_displacement = np.linspace(0.0, parameters.draw_length, force_draw_points)
    draw_force = np.zeros_like(draw_displacement)
    state = braced_state.copy()
    residual = braced_residual
    fixed = {int(index): float(state[index]) for index in clamps}
    fixed[midpoint_y_index] = 0.0
    for index, draw in enumerate(draw_displacement):
        prescribed_x = brace_x - float(draw)
        fixed[midpoint_x_index] = prescribed_x
        fixed[parameters.arrow_x_index] = prescribed_x
        state[midpoint_x_index] = prescribed_x
        state[midpoint_y_index] = 0.0
        state[parameters.arrow_x_index] = prescribed_x
        state, _, residual = solve_static_equilibrium(state, parameters, fixed)
        _, gradient, _, _ = distributed_energy_derivatives(state, parameters)
        draw_force[index] = abs(gradient[midpoint_x_index])
        state[parameters.arrow_x_index] = (
            prescribed_x - draw_force[index] / parameters.contact_stiffness
        )

    return DistributedStaticEquilibria(
        braced_state=braced_state,
        full_draw_state=state,
        brace_x=brace_x,
        draw_displacement=draw_displacement,
        draw_force=draw_force,
        braced_residual=braced_residual,
        full_draw_residual=residual,
    )


def implicit_step(
    state_old: np.ndarray,
    velocity_old: np.ndarray,
    parameters: DistributedBowParameters,
    mass: np.ndarray,
    damping: np.ndarray,
    fixed_values: dict[int, float],
    tolerance: float = 1.0e-6,
    maximum_iterations: int = 45,
) -> tuple[np.ndarray, np.ndarray, int]:
    dt = parameters.launch_dt
    state = state_old + dt * velocity_old
    fixed_indices = np.array(sorted(fixed_values), dtype=int)
    free_indices = np.setdiff1d(np.arange(parameters.degree_count), fixed_indices)

    def impose_fixed(candidate: np.ndarray) -> None:
        for index, value in fixed_values.items():
            candidate[index] = value

    impose_fixed(state)
    for iteration in range(1, maximum_iterations + 1):
        _, gradient, hessian, _ = distributed_energy_derivatives(state, parameters)
        velocity = (state - state_old) / dt
        residual = (
            mass * (velocity - velocity_old) / dt
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

    raise RuntimeError(
        "dynamic Newton solve did not converge; "
        f"residual={np.linalg.norm(residual[free_indices], ord=np.inf):.3e}"
    )


def run_distributed_bow(
    parameters: DistributedBowParameters | None = None,
    *,
    force_draw_points: int = 17,
    compute_flight: bool = True,
    verbose: bool = True,
) -> DistributedBowResult:
    parameters = parameters or DistributedBowParameters()
    equilibria = solve_equilibria(parameters, force_draw_points)
    mass, damping = mass_and_damping(parameters)
    state = equilibria.full_draw_state.copy()
    velocity = np.zeros(parameters.degree_count)
    fixed_indices = fixed_clamp_indices(parameters)
    fixed = {int(index): float(state[index]) for index in fixed_indices}
    midpoint_y_index = string_dof_indices(parameters)[parameters.string_midpoint, 1]
    fixed[midpoint_y_index] = 0.0

    step_count = int(round(parameters.launch_duration / parameters.launch_dt))
    time = np.arange(step_count + 1) * parameters.launch_dt
    states = [state.copy()]
    velocities = np.zeros((step_count + 1, parameters.degree_count))
    contact_force = np.zeros(step_count + 1)
    contact_gap = np.zeros(step_count + 1)
    arrow_speed = np.zeros(step_count + 1)
    maximum_limb_strain = np.zeros(step_count + 1)
    maximum_string_strain = np.zeros(step_count + 1)
    limb_energy = np.zeros(step_count + 1)
    string_energy = np.zeros(step_count + 1)
    contact_energy = np.zeros(step_count + 1)
    kinetic_energy = np.zeros(step_count + 1)
    arrow_kinetic = np.zeros(step_count + 1)
    damping_work = np.zeros(step_count + 1)
    energy_balance_error = np.zeros(step_count + 1)

    _, _, _, details = distributed_energy_derivatives(state, parameters)
    limb_energy[0] = float(details["limb_energy"])
    string_energy[0] = float(details["string_energy"])
    contact_energy[0] = float(details["contact_energy"])
    contact_force[0] = float(details["contact_force"])
    contact_gap[0] = float(details["contact_gap"])
    maximum_limb_strain[0] = float(details["maximum_limb_strain"])
    maximum_string_strain[0] = float(details["maximum_string_strain"])
    initial_energy = limb_energy[0] + string_energy[0] + contact_energy[0]
    braced_details = distributed_energy_derivatives(
        equilibria.braced_state, parameters
    )[3]
    braced_energy = float(
        braced_details["limb_energy"] + braced_details["string_energy"]
    )
    available_energy = initial_energy - braced_energy
    if available_energy <= 0.0:
        raise RuntimeError("full draw must store more energy than the braced state")
    maximum_newton_iterations = 0

    for step in range(step_count):
        state_old = state.copy()
        velocity_old = velocity.copy()
        state, velocity, iterations = implicit_step(
            state_old,
            velocity_old,
            parameters,
            mass,
            damping,
            fixed,
        )
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)
        states.append(state.copy())
        velocities[step + 1] = velocity
        _, _, _, details = distributed_energy_derivatives(state, parameters)
        limb_energy[step + 1] = float(details["limb_energy"])
        string_energy[step + 1] = float(details["string_energy"])
        contact_energy[step + 1] = float(details["contact_energy"])
        contact_force[step + 1] = float(details["contact_force"])
        contact_gap[step + 1] = float(details["contact_gap"])
        maximum_limb_strain[step + 1] = float(
            details["maximum_limb_strain"]
        )
        maximum_string_strain[step + 1] = float(
            details["maximum_string_strain"]
        )
        arrow_speed[step + 1] = velocity[parameters.arrow_x_index]
        kinetic_energy[step + 1] = 0.5 * np.sum(
            mass[:-1] * velocity[:-1] ** 2
        )
        arrow_kinetic[step + 1] = (
            0.5 * parameters.arrow_mass * velocity[parameters.arrow_x_index] ** 2
        )
        damping_work[step + 1] = damping_work[step] + parameters.launch_dt * np.sum(
            damping * velocity**2
        )
        accounted = (
            limb_energy[step + 1]
            + string_energy[step + 1]
            + contact_energy[step + 1]
            + kinetic_energy[step + 1]
            + arrow_kinetic[step + 1]
            + damping_work[step + 1]
        )
        energy_balance_error[step + 1] = initial_energy - accounted

    active = contact_force > parameters.contact_force_tolerance
    separations = np.flatnonzero(active[:-1] & ~active[1:])
    if not len(separations):
        raise RuntimeError("final arrow separation was not detected")
    separation_index = int(separations[-1])
    gap0 = contact_gap[separation_index]
    gap1 = contact_gap[separation_index + 1]
    denominator = gap1 - gap0
    fraction = -gap0 / denominator if denominator > 0.0 else 1.0
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
    release_speed = (
        velocities[separation_index, parameters.arrow_x_index]
        + fraction
        * (
            velocities[separation_index + 1, parameters.arrow_x_index]
            - velocities[separation_index, parameters.arrow_x_index]
        )
    )
    release_force_before = contact_force[separation_index]
    release_force_after = contact_force[separation_index + 1]
    launch_efficiency = (
        0.5 * parameters.arrow_mass * release_speed**2 / available_energy
    )
    draw_force_constraint_satisfied = bool(
        np.max(equilibria.draw_force) <= parameters.maximum_draw_force
    )
    limb_strain_constraint_satisfied = bool(
        np.max(maximum_limb_strain)
        <= parameters.maximum_limb_axial_strain
    )
    string_strain_constraint_satisfied = bool(
        np.max(maximum_string_strain) <= parameters.maximum_string_strain
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
        print("Distributed DER straight-end bow simulation")
        print(
            f"  limb/string nodes          : "
            f"{parameters.limb_node_count} per limb / "
            f"{parameters.string_node_count}"
        )
        print(f"  brace-plane x              : {equilibria.brace_x:.6f} m")
        print(f"  full-draw force            : {equilibria.draw_force[-1]:.3f} N")
        print(f"  initial stored energy      : {initial_energy:.3f} J")
        print(f"  braced preload energy      : {braced_energy:.3f} J")
        print(f"  available draw energy      : {available_energy:.3f} J")
        print(f"  final release time         : {1e3 * release_time:.3f} ms")
        print(f"  final release x            : {release_position:.6f} m")
        print(f"  release speed              : {release_speed:.3f} m/s")
        print(f"  launch efficiency          : {100.0 * launch_efficiency:.2f}%")
        print(
            f"  maximum limb axial strain  : "
            f"{100.0 * np.max(maximum_limb_strain):.4f}%"
        )
        print(
            f"  maximum string strain      : "
            f"{100.0 * np.max(maximum_string_strain):.3f}%"
        )
        print(
            f"  design constraints         : "
            f"draw={'PASS' if draw_force_constraint_satisfied else 'FAIL'}, "
            f"limb={'PASS' if limb_strain_constraint_satisfied else 'FAIL'}, "
            f"string={'PASS' if string_strain_constraint_satisfied else 'FAIL'}"
        )
        print(f"  maximum Newton iterations  : {maximum_newton_iterations}")
        print(f"  numerical energy loss      : {energy_balance_error[-1]:.6f} J")

    return DistributedBowResult(
        parameters=parameters,
        time=time,
        states=states,
        velocities=velocities,
        contact_force=contact_force,
        contact_gap=contact_gap,
        arrow_speed=arrow_speed,
        maximum_limb_strain=maximum_limb_strain,
        maximum_string_strain=maximum_string_strain,
        limb_energy=limb_energy,
        string_energy=string_energy,
        contact_energy=contact_energy,
        kinetic_energy=kinetic_energy,
        arrow_kinetic_energy=arrow_kinetic,
        damping_work=damping_work,
        energy_balance_error=energy_balance_error,
        release_time=float(release_time),
        release_position=float(release_position),
        release_speed=float(release_speed),
        release_force_before=float(release_force_before),
        release_force_after=float(release_force_after),
        initial_energy=float(initial_energy),
        braced_energy=braced_energy,
        available_energy=available_energy,
        launch_efficiency=float(launch_efficiency),
        draw_force_constraint_satisfied=draw_force_constraint_satisfied,
        limb_strain_constraint_satisfied=limb_strain_constraint_satisfied,
        string_strain_constraint_satisfied=string_strain_constraint_satisfied,
        maximum_newton_iterations=maximum_newton_iterations,
        equilibria=equilibria,
        required_aim_angle=float(aim_angle),
        flight_time=flight_time,
        flight_position=flight_position,
        flight_velocity=flight_velocity,
        target_time=float(target_time),
        target_speed=float(target_speed),
        target_height_error=float(target_height_error),
    )


def plot_distributed_result(result: DistributedBowResult) -> plt.Figure:
    """Plot distributed geometry, contact, energy, and force-draw response."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    snapshot_times = [0.0, result.release_time, 0.025, result.time[-1]]
    labels = ["full draw", "final release", "post-release", "late response"]
    natural_upper, natural_lower = natural_limb_nodes(result.parameters)
    axes[0, 0].plot(
        natural_upper[1:, 0],
        natural_upper[1:, 1],
        "--",
        color="0.65",
        linewidth=1.5,
        label="unstrung natural shape",
    )
    axes[0, 0].plot(
        natural_lower[1:, 0],
        natural_lower[1:, 1],
        "--",
        color="0.65",
        linewidth=1.5,
    )
    for target_time, label in zip(snapshot_times, labels):
        index = int(np.argmin(np.abs(result.time - target_time)))
        details = distributed_energy_derivatives(
            result.states[index], result.parameters
        )[3]
        upper = details["upper_nodes"]
        lower = details["lower_nodes"]
        string = details["string_nodes"]
        axes[0, 0].plot(upper[1:, 0], upper[1:, 1], "o-", ms=2)
        axes[0, 0].plot(lower[1:, 0], lower[1:, 1], "o-", ms=2)
        axes[0, 0].plot(string[:, 0], string[:, 1], "o-", ms=2, label=label)
    axes[0, 0].plot(
        [0.0, 0.0],
        [-result.parameters.riser_half_length, result.parameters.riser_half_length],
        color="0.15",
        linewidth=6,
        label="rigid riser",
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("shooting direction x [m]")
    axes[0, 0].set_ylabel("vertical position y [m]")
    axes[0, 0].set_title("Distributed DER straight-end bow")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(result.time * 1e3, result.arrow_speed, label="arrow speed")
    axes[0, 1].set_xlabel("time [ms]")
    axes[0, 1].set_ylabel("arrow speed [m/s]")
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
    axes[0, 1].set_title("Unilateral nock contact")

    axes[1, 0].plot(result.time * 1e3, result.limb_energy, label="limb elastic")
    axes[1, 0].plot(result.time * 1e3, result.string_energy, label="string elastic")
    axes[1, 0].plot(result.time * 1e3, result.kinetic_energy, label="bow/string kinetic")
    axes[1, 0].plot(result.time * 1e3, result.arrow_kinetic_energy, label="arrow kinetic")
    axes[1, 0].plot(result.time * 1e3, result.damping_work, label="damping work")
    axes[1, 0].set_xlabel("time [ms]")
    axes[1, 0].set_ylabel("energy [J]")
    axes[1, 0].set_title("Energy partition")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        result.equilibria.draw_displacement,
        result.equilibria.draw_force,
        "o-",
        label="distributed bow",
    )
    axes[1, 1].plot(
        [0.429],
        [126.9],
        "s",
        color="tab:red",
        label="Tuijn--Kooi endpoint",
    )
    axes[1, 1].set_xlabel("draw displacement [m]")
    axes[1, 1].set_ylabel("draw force [N]")
    axes[1, 1].set_title("Force-draw curve")
    axes[1, 1].legend()
    figure.tight_layout()
    return figure


def animate_distributed_bow(result: DistributedBowResult) -> FuncAnimation:
    """Animate the distributed bow, DER string, and independent arrow."""
    parameters = result.parameters
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.set_aspect("equal")
    axis.set_xlim(-0.75, 2.5)
    axis.set_ylim(-1.0, 1.0)
    axis.set_xlabel("shooting direction x [m]")
    axis.set_ylabel("vertical position y [m]")
    axis.set_title("Distributed DER bow launch")
    axis.plot(
        [0.0, 0.0],
        [-parameters.riser_half_length, parameters.riser_half_length],
        color="0.15",
        linewidth=7,
        solid_capstyle="round",
        label="rigid riser",
    )
    natural_upper, natural_lower = natural_limb_nodes(parameters)
    axis.plot(
        natural_upper[1:, 0],
        natural_upper[1:, 1],
        "--",
        color="0.70",
        linewidth=1.2,
        label="unstrung natural shape",
    )
    axis.plot(
        natural_lower[1:, 0],
        natural_lower[1:, 1],
        "--",
        color="0.70",
        linewidth=1.2,
    )
    upper_line, = axis.plot([], [], "o-", color="saddlebrown", ms=3, lw=3)
    lower_line, = axis.plot([], [], "o-", color="saddlebrown", ms=3, lw=3)
    string_line, = axis.plot([], [], "o-", color="steelblue", ms=3, lw=2)
    arrow_line, = axis.plot([], [], color="darkorange", lw=3)
    arrow_head, = axis.plot([], [], ">", color="darkorange", ms=8)
    status = axis.text(0.02, 0.97, "", transform=axis.transAxes, va="top")
    axis.legend(loc="lower right")
    frame_count = min(241, len(result.time))
    frame_indices = np.unique(
        np.rint(np.linspace(0, len(result.time) - 1, frame_count)).astype(int)
    )

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        state = result.states[index]
        details = distributed_energy_derivatives(state, parameters)[3]
        upper = details["upper_nodes"]
        lower = details["lower_nodes"]
        string = details["string_nodes"]
        upper_line.set_data(upper[1:, 0], upper[1:, 1])
        lower_line.set_data(lower[1:, 0], lower[1:, 1])
        string_line.set_data(string[:, 0], string[:, 1])
        arrow_x = state[parameters.arrow_x_index]
        arrow_tip = arrow_x + parameters.arrow_length
        arrow_line.set_data([arrow_x, arrow_tip], [0.0, 0.0])
        arrow_head.set_data([arrow_tip], [0.0])
        if result.contact_force[index] > parameters.contact_force_tolerance:
            phase = "string-arrow contact active"
        elif result.time[index] < result.release_time:
            phase = "temporary separation before recontact"
        else:
            phase = "final separation; distributed bow vibration"
        status.set_text(
            f"t = {1e3 * result.time[index]:5.2f} ms   "
            f"v = {result.arrow_speed[index]:5.2f} m/s   "
            f"F_contact = {result.contact_force[index]:6.2f} N\n{phase}"
        )
        return upper_line, lower_line, string_line, arrow_line, arrow_head, status

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
    parser.add_argument("--limb-segments", type=int, default=10)
    parser.add_argument("--string-nodes", type=int, default=21)
    parser.add_argument("--limb-EI", type=float, default=68.0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument(
        "--save-animation",
        type=Path,
        help="save the launch animation as .mp4 or .gif",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        help="save the four-panel result figure",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    parameters = DistributedBowParameters(
        limb_segments=args.limb_segments,
        string_node_count=args.string_nodes,
        limb_EI=args.limb_EI,
    )
    result = run_distributed_bow(parameters, verbose=not args.quiet)
    if args.save_figure is not None:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        figure = plot_distributed_result(result)
        figure.savefig(args.save_figure, dpi=180, bbox_inches="tight")
        plt.close(figure)
    if args.save_animation is not None:
        args.save_animation.parent.mkdir(parents=True, exist_ok=True)
        animation = animate_distributed_bow(result)
        suffix = args.save_animation.suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        animation.save(args.save_animation, writer=writer, fps=30, dpi=120)
        plt.close(animation._fig)
    if not args.no_show:
        if args.animate:
            animation = animate_distributed_bow(result)
            _ = animation
        else:
            plot_distributed_result(result)
        plt.show()


if __name__ == "__main__":
    main()
