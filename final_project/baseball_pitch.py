"""Pitcher force, free flight, and impact of a baseball-sized soft ball.

Course methods:
  * discrete elastic shell stretching and bending,
  * backward Euler time integration,
  * Newton's method,
  * high-Reynolds-number quadratic drag.

Project extensions:
  * a smooth, localized pitcher-force pulse,
  * a zero-net-force pitcher torque and visible material marker,
  * a regulation-size, regulation-mass baseball approximation,
  * a center mass and radial elastic core for shape recovery,
  * area-weighted smooth wall and ground contact with local substepping,
  * a tracking-camera animation for the long flight distance.

The model is intentionally more compliant than an actual baseball: it is a
baseball-inspired soft robotic body, represented by an effective shell surrounding
an elastic radial core.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from soft_shell_collision import (
    COURSE_SHELL_DIR,
    make_icosphere,
    plane_contact,
    triangle_lumped_masses,
)

sys.path.insert(0, str(COURSE_SHELL_DIR))
from shell_functions import getEdgesAndHinges, getFb, getFs, getThetaBar  # noqa: E402


@dataclass
class BaseballResult:
    time: np.ndarray
    states: list[np.ndarray]
    stored_steps: np.ndarray
    center_of_mass: np.ndarray
    center_velocity: np.ndarray
    diameter: np.ndarray
    radial_dent: np.ndarray
    pitcher_force: np.ndarray
    spin_torque: np.ndarray
    twist_angle: np.ndarray
    drag_force: np.ndarray
    wall_force: np.ndarray
    ground_force: np.ndarray
    wall_penetration: np.ndarray
    ground_penetration: np.ndarray
    stretching_force: np.ndarray
    bending_force: np.ndarray
    core_force: np.ndarray
    elements: np.ndarray
    wall_x: float
    radius: float
    mass: float
    throw_duration: float
    wall_hit_time: float | None
    ground_hit_time: float | None
    release_speed: float
    rebound_speed: float | None
    impact_speed: float | None
    coefficient_of_restitution: float | None
    wall_contact_duration: float | None
    wall_compression: float | None
    wall_diameter_recovery: float | None
    wall_recovery_time: float | None
    residual_radial_dent: float | None
    marker_node: int


def pitcher_patch_weights(nodes: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Concentrate the hand force smoothly on the rear side of the ball."""
    local_x = (nodes[:, 0] - center[0]) / radius
    # A Gaussian centered at the rear pole avoids a mesh-dependent hard cutoff.
    weights = np.exp(-((local_x + 1.0) / 0.45) ** 2)
    weights /= weights.sum()
    return weights


def pitcher_force(
    time: float,
    duration: float,
    mass: float,
    target_velocity: np.ndarray,
    gravity: np.ndarray,
    patch_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return a half-sine force pulse whose impulse gives the target velocity."""
    force = np.zeros(3 * len(patch_weights))
    if time < 0.0 or time > duration:
        return force, 0.0

    # Compensate the gravitational impulse accumulated during the throw.  Drag is
    # small during this short interval and is left for the solver to apply.
    required_impulse = mass * (target_velocity - gravity * duration)
    peak_force = np.pi * required_impulse / (2.0 * duration)
    total_force = peak_force * np.sin(np.pi * time / duration)
    force.reshape(-1, 3)[:] = patch_weights[:, None] * total_force[None, :]
    return force, float(np.linalg.norm(total_force))


def pitcher_spin_force(
    time: float,
    duration: float,
    target_angular_velocity: np.ndarray,
    nodal_mass: np.ndarray,
    reference_offsets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply a zero-net-force torque pulse that builds baseball backspin.

    The half-sine angular-acceleration pulse is the rotational analogue of the
    translational pitcher-force pulse.  Forces are distributed by nodal mass as
    ``m_i * (alpha x r_i)`` and corrected to have exactly zero resultant force.
    """
    force = np.zeros_like(reference_offsets)
    if time < 0.0 or time > duration:
        return force.reshape(-1), 0.0

    peak_angular_acceleration = (
        np.pi * target_angular_velocity / (2.0 * duration)
    )
    angular_acceleration = peak_angular_acceleration * np.sin(
        np.pi * time / duration
    )
    force[:] = nodal_mass[:, None] * np.cross(
        angular_acceleration[None, :], reference_offsets
    )
    force -= (
        nodal_mass[:, None]
        / nodal_mass.sum()
        * force.sum(axis=0)[None, :]
    )
    torque = np.sum(np.cross(reference_offsets, force), axis=0)
    return force.reshape(-1), float(np.linalg.norm(torque))


def sphere_drag(
    nodal_velocity: np.ndarray,
    velocity_position_factor: float,
    nodal_weights: np.ndarray,
    radius: float,
    air_density: float,
    drag_coefficient: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Quadratic sphere drag and its backward-Euler position Jacobian."""
    center_velocity = np.sum(nodal_weights[:, None] * nodal_velocity, axis=0)
    speed = np.linalg.norm(center_velocity)
    ndof = 3 * len(nodal_weights)
    force = np.zeros(ndof)
    jacobian = np.zeros((ndof, ndof))
    if speed < 1.0e-12:
        return force, jacobian, 0.0

    area = np.pi * radius**2
    coefficient = 0.5 * air_density * drag_coefficient * area
    total_drag = -coefficient * speed * center_velocity
    force.reshape(-1, 3)[:] = nodal_weights[:, None] * total_drag[None, :]

    derivative_velocity = -coefficient * (
        speed * np.eye(3) + np.outer(center_velocity, center_velocity) / speed
    )
    for i, weight_i in enumerate(nodal_weights):
        rows = slice(3 * i, 3 * i + 3)
        for j, weight_j in enumerate(nodal_weights):
            cols = slice(3 * j, 3 * j + 3)
            jacobian[rows, cols] = (
                weight_i
                * weight_j
                * velocity_position_factor
                * derivative_velocity
            )
    return force, jacobian, float(np.linalg.norm(total_drag))


def baseball_contact(
    q: np.ndarray,
    velocity_old: np.ndarray,
    wall_x: float,
    stiffness: float | np.ndarray,
    damping: float,
    surface_node_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Penalty contact with damping evaluated from the known old velocity.

    Keeping the dashpot explicit avoids a velocity-sign switch inside the Newton
    solve while the elastic penalty stiffness remains implicit.
    """
    force = np.zeros_like(q)
    jacobian = np.zeros((len(q), len(q)))
    wall_total = 0.0
    ground_total = 0.0
    wall_max_penetration = 0.0
    ground_max_penetration = 0.0
    smoothing_length = 1.0e-5
    for node in range(surface_node_count):
        node_stiffness = (
            float(stiffness[node]) if np.ndim(stiffness) else float(stiffness)
        )
        ix = 3 * node
        iz = ix + 2
        wall_penetration = q[ix] - wall_x
        wall_root = np.hypot(wall_penetration, smoothing_length)
        wall_positive_part = 0.5 * (wall_penetration + wall_root)
        wall_positive_gradient = 0.5 * (
            1.0 + wall_penetration / wall_root
        )
        contact_force = -node_stiffness * wall_positive_part
        if wall_penetration > 0.0:
            contact_force -= damping * max(velocity_old[ix], 0.0)
            wall_total += -contact_force
            wall_max_penetration = max(wall_max_penetration, wall_penetration)
        force[ix] += contact_force
        jacobian[ix, ix] -= node_stiffness * wall_positive_gradient

        ground_penetration = -q[iz]
        ground_root = np.hypot(ground_penetration, smoothing_length)
        ground_positive_part = 0.5 * (ground_penetration + ground_root)
        ground_positive_gradient = 0.5 * (
            1.0 + ground_penetration / ground_root
        )
        contact_force = node_stiffness * ground_positive_part
        if ground_penetration > 0.0:
            contact_force -= damping * min(velocity_old[iz], 0.0)
            ground_total += contact_force
            ground_max_penetration = max(
                ground_max_penetration, ground_penetration
            )
        force[iz] += contact_force
        jacobian[iz, iz] -= node_stiffness * ground_positive_gradient
    return (
        force,
        jacobian,
        wall_total,
        ground_total,
        wall_max_penetration,
        ground_max_penetration,
    )


def maximum_core_spring_force(
    q: np.ndarray,
    edges: np.ndarray,
    reference_lengths: np.ndarray,
    stretching_stiffness: np.ndarray,
    core_edge_start: int,
) -> float:
    """Maximum axial force EA*(length/rest_length - 1) in a core spring."""
    if core_edge_start >= len(edges):
        return 0.0
    positions = q.reshape(-1, 3)
    core_edges = edges[core_edge_start:]
    lengths = np.linalg.norm(
        positions[core_edges[:, 1]] - positions[core_edges[:, 0]], axis=1
    )
    strain = lengths / reference_lengths[core_edge_start:] - 1.0
    return float(
        np.max(np.abs(stretching_stiffness[core_edge_start:] * strain))
    )


def implicit_step(
    *,
    q_old: np.ndarray,
    velocity_old: np.ndarray,
    time_new: float,
    dt: float,
    mass_vector: np.ndarray,
    nodal_weights: np.ndarray,
    edges: np.ndarray,
    reference_lengths: np.ndarray,
    stretching_stiffness: np.ndarray,
    core_edge_start: int,
    surface_node_count: int,
    hinges: np.ndarray,
    natural_angles: np.ndarray,
    bending_stiffness: float,
    gravity_force: np.ndarray,
    throw_duration: float,
    total_mass: float,
    target_velocity: np.ndarray,
    gravity: np.ndarray,
    pitcher_weights: np.ndarray,
    spin_duration: float,
    target_angular_velocity: np.ndarray,
    spin_reference_offsets: np.ndarray,
    radius: float,
    air_density: float,
    drag_coefficient: float,
    wall_x: float,
    contact_stiffness: float | np.ndarray,
    contact_damping: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], int]:
    q = q_old + dt * velocity_old
    mass_over_dt2 = mass_vector / dt**2
    hand_force, hand_force_magnitude = pitcher_force(
        time_new,
        throw_duration,
        total_mass,
        target_velocity,
        gravity,
        pitcher_weights,
    )
    spin_force, spin_torque_magnitude = pitcher_spin_force(
        time_new,
        spin_duration,
        target_angular_velocity,
        mass_vector.reshape(-1, 3)[:, 0],
        spin_reference_offsets,
    )

    for iteration in range(1, maximum_iterations + 1):
        bending_force, bending_jacobian = getFb(
            q, hinges, natural_angles, bending_stiffness
        )
        stretching_force, stretching_jacobian = getFs(
            q, edges, reference_lengths, stretching_stiffness
        )
        drag_force, drag_jacobian, drag_magnitude = sphere_drag(
            (q.reshape(-1, 3) - q_old.reshape(-1, 3)) / dt,
            1.0 / dt,
            nodal_weights,
            radius,
            air_density,
            drag_coefficient,
        )
        (
            contact_force,
            contact_jacobian,
            wall_force,
            ground_force,
            wall_penetration,
            ground_penetration,
        ) = baseball_contact(
            q,
            velocity_old,
            wall_x,
            contact_stiffness,
            contact_damping,
            surface_node_count,
        )

        total_force = (
            bending_force
            + stretching_force
            + gravity_force
            + hand_force
            + spin_force
            + drag_force
            + contact_force
        )
        force_jacobian = (
            bending_jacobian
            + stretching_jacobian
            + drag_jacobian
            + contact_jacobian
        )
        residual = mass_vector * (
            (q - q_old) / dt**2 - velocity_old / dt
        ) - total_force
        error = np.linalg.norm(residual, ord=np.inf)
        if error <= tolerance:
            break

        system_matrix = np.diag(mass_over_dt2) - force_jacobian
        q -= np.linalg.solve(system_matrix, residual)
    else:
        raise RuntimeError(
            f"Newton did not converge at t={time_new:.6f} s; "
            f"residual={error:.3e} N"
        )

    velocity = (q - q_old) / dt
    maximum_stretching_force = float(
        np.linalg.norm(stretching_force.reshape(-1, 3), axis=1).max()
    )
    maximum_bending_force = float(
        np.linalg.norm(bending_force.reshape(-1, 3), axis=1).max()
    )
    maximum_core_force = maximum_core_spring_force(
        q,
        edges,
        reference_lengths,
        stretching_stiffness,
        core_edge_start,
    )
    diagnostics = (
        hand_force_magnitude,
        spin_torque_magnitude,
        drag_magnitude,
        wall_force,
        ground_force,
        wall_penetration,
        ground_penetration,
        maximum_stretching_force,
        maximum_bending_force,
        maximum_core_force,
    )
    return q, velocity, diagnostics, iteration


def newmark_step(
    *,
    q_old: np.ndarray,
    velocity_old: np.ndarray,
    acceleration_old: np.ndarray,
    time_new: float,
    dt: float,
    mass_vector: np.ndarray,
    nodal_weights: np.ndarray,
    edges: np.ndarray,
    reference_lengths: np.ndarray,
    stretching_stiffness: np.ndarray,
    core_edge_start: int,
    surface_node_count: int,
    hinges: np.ndarray,
    natural_angles: np.ndarray,
    bending_stiffness: float,
    gravity_force: np.ndarray,
    throw_duration: float,
    total_mass: float,
    target_velocity: np.ndarray,
    gravity: np.ndarray,
    pitcher_weights: np.ndarray,
    spin_duration: float,
    target_angular_velocity: np.ndarray,
    spin_reference_offsets: np.ndarray,
    radius: float,
    air_density: float,
    drag_coefficient: float,
    wall_x: float,
    contact_stiffness: float | np.ndarray,
    contact_damping: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...], int]:
    """Average-acceleration Newmark step (beta=1/4, gamma=1/2)."""
    beta = 0.25
    gamma = 0.50
    q_predictor = (
        q_old
        + dt * velocity_old
        + dt**2 * (0.5 - beta) * acceleration_old
    )
    velocity_predictor = velocity_old + dt * (1.0 - gamma) * acceleration_old
    q = q_predictor + beta * dt**2 * acceleration_old
    inertia_factor = 1.0 / (beta * dt**2)
    velocity_position_factor = gamma / (beta * dt)
    hand_force, hand_force_magnitude = pitcher_force(
        time_new,
        throw_duration,
        total_mass,
        target_velocity,
        gravity,
        pitcher_weights,
    )
    spin_force, spin_torque_magnitude = pitcher_spin_force(
        time_new,
        spin_duration,
        target_angular_velocity,
        mass_vector.reshape(-1, 3)[:, 0],
        spin_reference_offsets,
    )

    for iteration in range(1, maximum_iterations + 1):
        acceleration = inertia_factor * (q - q_predictor)
        velocity = velocity_predictor + gamma * dt * acceleration
        bending_force, bending_jacobian = getFb(
            q, hinges, natural_angles, bending_stiffness
        )
        stretching_force, stretching_jacobian = getFs(
            q, edges, reference_lengths, stretching_stiffness
        )
        drag_force, drag_jacobian, drag_magnitude = sphere_drag(
            velocity.reshape(-1, 3),
            velocity_position_factor,
            nodal_weights,
            radius,
            air_density,
            drag_coefficient,
        )
        (
            contact_force,
            contact_jacobian,
            wall_force,
            ground_force,
            wall_penetration,
            ground_penetration,
        ) = baseball_contact(
            q,
            velocity_old,
            wall_x,
            contact_stiffness,
            contact_damping,
            surface_node_count,
        )

        total_force = (
            bending_force
            + stretching_force
            + gravity_force
            + hand_force
            + spin_force
            + drag_force
            + contact_force
        )
        force_jacobian = (
            bending_jacobian
            + stretching_jacobian
            + drag_jacobian
            + contact_jacobian
        )
        residual = mass_vector * acceleration - total_force
        error = np.linalg.norm(residual, ord=np.inf)
        if error <= tolerance:
            break

        system_matrix = np.diag(mass_vector * inertia_factor) - force_jacobian
        # Contact activation is piecewise smooth. A relaxed Newton update avoids
        # bouncing between the active and inactive contact sets at first impact.
        relaxation = 0.5 if (wall_force > 0.0 or ground_force > 0.0) else 1.0
        q -= relaxation * np.linalg.solve(system_matrix, residual)
    else:
        raise RuntimeError(
            f"Newmark Newton solve did not converge at t={time_new:.6f} s; "
            f"residual={error:.3e} N"
        )

    acceleration = inertia_factor * (q - q_predictor)
    velocity = velocity_predictor + gamma * dt * acceleration
    maximum_stretching_force = float(
        np.linalg.norm(stretching_force.reshape(-1, 3), axis=1).max()
    )
    maximum_bending_force = float(
        np.linalg.norm(bending_force.reshape(-1, 3), axis=1).max()
    )
    maximum_core_force = maximum_core_spring_force(
        q,
        edges,
        reference_lengths,
        stretching_stiffness,
        core_edge_start,
    )
    diagnostics = (
        hand_force_magnitude,
        spin_torque_magnitude,
        drag_magnitude,
        wall_force,
        ground_force,
        wall_penetration,
        ground_penetration,
        maximum_stretching_force,
        maximum_bending_force,
        maximum_core_force,
    )
    return q, velocity, acceleration, diagnostics, iteration


def run_baseball(
    *,
    subdivisions: int = 1,
    dt: float = 0.001,
    total_time: float = 1.15,
    output_stride: int = 4,
    effective_youngs_modulus: float = 8.0e7,
    bending_multiplier: float = 6.0,
    core_total_stiffness: float = 1.0e5,
    target_spin_rpm: float = 1200.0,
    contact_stiffness: float = 1.0e6,
    contact_damping: float = 0.0,
    verbose: bool = True,
) -> BaseballResult:
    # Regulation midpoint values: 5--5.25 oz and 9--9.25 in circumference.
    mass = 0.1453
    circumference = 0.5 * (9.0 + 9.25) * 0.0254
    radius = circumference / (2.0 * np.pi)
    center0 = np.array([0.0, 0.0, 1.80])
    wall_x = 18.44  # 60 ft 6 in

    # Effective shell properties. They are chosen to give millimeter-scale
    # impact deformation; they are not literal properties of leather or yarn.
    effective_thickness = 0.0025
    target_velocity = np.array([40.0, 0.0, 1.5])
    throw_duration = 0.080
    spin_duration = 0.020
    target_angular_velocity = np.array(
        [0.0, -2.0 * np.pi * target_spin_rpm / 60.0, 0.0]
    )
    gravity = np.array([0.0, 0.0, -9.81])
    air_density = 1.225
    drag_coefficient = 0.35
    tolerance = 1.0e-4
    maximum_iterations = 60

    surface_nodes0, elements = make_icosphere(radius, center0, subdivisions)
    surface_node_count = len(surface_nodes0)
    center_node_index = surface_node_count
    nodes0 = np.vstack((surface_nodes0, center0))
    shell_edges, hinges = getEdgesAndHinges(elements)
    core_edges = np.column_stack(
        (np.arange(surface_node_count), np.full(surface_node_count, center_node_index))
    )
    core_edge_start = len(shell_edges)
    edges = np.vstack((shell_edges, core_edges))
    q = nodes0.reshape(-1).copy()
    velocity = np.zeros_like(q)

    reference_lengths = np.linalg.norm(
        nodes0[edges[:, 1]] - nodes0[edges[:, 0]], axis=1
    )
    shell_stretching_stiffness = (
        np.sqrt(3.0) / 2.0
        * effective_youngs_modulus
        * effective_thickness
        * reference_lengths[:core_edge_start]
    )
    # The total radial stiffness is kept mesh-independent by dividing it among
    # all surface-to-center springs.  EA=k*rest_length matches getFs units.
    core_linear_stiffness = core_total_stiffness / surface_node_count
    core_stretching_stiffness = (
        core_linear_stiffness * reference_lengths[core_edge_start:]
    )
    stretching_stiffness = np.concatenate(
        (shell_stretching_stiffness, core_stretching_stiffness)
    )
    bending_stiffness = (
        bending_multiplier
        *
        2.0
        / np.sqrt(3.0)
        * effective_youngs_modulus
        * effective_thickness**3
        / 12.0
    )
    natural_angles = getThetaBar(q, hinges)

    raw_surface_mass = triangle_lumped_masses(
        surface_nodes0, elements, 1.0, 1.0
    )
    surface_weights = raw_surface_mass / raw_surface_mass.sum()
    # Area-weighted penalty stiffness keeps the integrated contact response
    # approximately mesh-independent.  The factor 12 preserves the original
    # coarse-icosahedron calibration.
    contact_nodal_stiffness = 12.0 * contact_stiffness * surface_weights
    shell_mass_fraction = 0.45
    core_mass_fraction = 1.0 - shell_mass_fraction
    nodal_mass = np.concatenate(
        (mass * shell_mass_fraction * surface_weights, [mass * core_mass_fraction])
    )
    nodal_weights = nodal_mass / mass
    mass_vector = np.repeat(nodal_mass, 3)
    gravity_force = mass_vector * np.tile(gravity, len(nodes0))
    hand_weights = np.concatenate(
        (pitcher_patch_weights(surface_nodes0, center0, radius), [0.0])
    )
    spin_reference_offsets = nodes0 - center0
    marker_node = int(
        np.argmax(
            surface_nodes0[:, 2] - center0[2]
            + 0.20 * (surface_nodes0[:, 0] - center0[0])
        )
    )

    number_of_steps = int(round(total_time / dt))
    time = np.arange(number_of_steps + 1) * dt
    center_of_mass = np.zeros((number_of_steps + 1, 3))
    center_velocity = np.zeros((number_of_steps + 1, 3))
    diameter = np.zeros((number_of_steps + 1, 3))
    radial_dent = np.zeros(number_of_steps + 1)
    pitcher_force_history = np.zeros(number_of_steps + 1)
    spin_torque_history = np.zeros(number_of_steps + 1)
    twist_angle = np.zeros(number_of_steps + 1)
    drag_force_history = np.zeros(number_of_steps + 1)
    wall_force = np.zeros(number_of_steps + 1)
    ground_force = np.zeros(number_of_steps + 1)
    wall_penetration = np.zeros(number_of_steps + 1)
    ground_penetration = np.zeros(number_of_steps + 1)
    stretching_force_history = np.zeros(number_of_steps + 1)
    bending_force_history = np.zeros(number_of_steps + 1)
    core_force_history = np.zeros(number_of_steps + 1)
    stored_steps = [0]
    states = [q.copy()]

    def record(step: int) -> None:
        positions = q.reshape(-1, 3)
        velocities = velocity.reshape(-1, 3)
        center_of_mass[step] = np.sum(
            nodal_weights[:, None] * positions, axis=0
        )
        center_velocity[step] = np.sum(
            nodal_weights[:, None] * velocities, axis=0
        )
        diameter[step] = np.ptp(positions[:surface_node_count], axis=0)
        radial_lengths = np.linalg.norm(
            positions[:surface_node_count] - positions[center_node_index], axis=1
        )
        radial_dent[step] = max(0.0, radius - radial_lengths.min())
        marker_offset = positions[marker_node] - center_of_mass[step]
        twist_angle[step] = np.arctan2(marker_offset[0], marker_offset[2])

    record(0)
    maximum_newton_iterations = 0
    for step in range(1, number_of_steps + 1):
        surface_positions = q.reshape(-1, 3)[:surface_node_count]
        near_wall = surface_positions[:, 0].max() > wall_x - 0.020
        near_ground = surface_positions[:, 2].min() < 0.020
        substep_count = 4 if (near_wall or near_ground) else 1
        substep_dt = dt / substep_count
        diagnostics_array = np.zeros(10)
        iterations = 0
        for substep in range(substep_count):
            substep_time = time[step - 1] + (substep + 1) * substep_dt
            q, velocity, substep_diagnostics, substep_iterations = implicit_step(
                q_old=q,
                velocity_old=velocity,
                time_new=substep_time,
                dt=substep_dt,
                mass_vector=mass_vector,
                nodal_weights=nodal_weights,
                edges=edges,
                reference_lengths=reference_lengths,
                stretching_stiffness=stretching_stiffness,
                core_edge_start=core_edge_start,
                surface_node_count=surface_node_count,
                hinges=hinges,
                natural_angles=natural_angles,
                bending_stiffness=bending_stiffness,
                gravity_force=gravity_force,
                throw_duration=throw_duration,
                total_mass=mass,
                target_velocity=target_velocity,
                gravity=gravity,
                pitcher_weights=hand_weights,
                spin_duration=spin_duration,
                target_angular_velocity=target_angular_velocity,
                spin_reference_offsets=spin_reference_offsets,
                radius=radius,
                air_density=air_density,
                drag_coefficient=drag_coefficient,
                wall_x=wall_x,
                contact_stiffness=contact_nodal_stiffness,
                contact_damping=contact_damping,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
            )
            diagnostics_array = np.maximum(
                diagnostics_array, np.asarray(substep_diagnostics)
            )
            iterations = max(iterations, substep_iterations)
        diagnostics = tuple(diagnostics_array)
        (
            pitcher_force_history[step],
            spin_torque_history[step],
            drag_force_history[step],
            wall_force[step],
            ground_force[step],
            wall_penetration[step],
            ground_penetration[step],
            stretching_force_history[step],
            bending_force_history[step],
            core_force_history[step],
        ) = diagnostics
        record(step)
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)

        if step % output_stride == 0 or step == number_of_steps:
            stored_steps.append(step)
            states.append(q.copy())
        if verbose and step % max(1, number_of_steps // 10) == 0:
            print(
                f"t={time[step]:.3f} s, x={center_of_mass[step, 0]:.3f} m, "
                f"z={center_of_mass[step, 2]:.3f} m, "
                f"speed={np.linalg.norm(center_velocity[step]):.2f} m/s"
            )

    twist_angle = np.unwrap(twist_angle)

    wall_indices = np.flatnonzero(wall_force > 1.0e-8)
    ground_indices = np.flatnonzero(ground_force > 1.0e-8)
    wall_hit_time = float(time[wall_indices[0]]) if wall_indices.size else None
    ground_hit_time = (
        float(time[ground_indices[0]]) if ground_indices.size else None
    )
    release_index = min(number_of_steps, int(round(throw_duration / dt)))
    release_speed = float(np.linalg.norm(center_velocity[release_index]))
    rebound_speed = None
    impact_speed = None
    coefficient_of_restitution = None
    wall_contact_duration = None
    wall_compression = None
    wall_diameter_recovery = None
    wall_recovery_time = None
    residual_radial_dent = None
    if wall_indices.size:
        first_wall = int(wall_indices[0])
        last_wall = int(wall_indices[-1])
        impact_index = max(0, first_wall - 1)
        impact_speed = float(abs(center_velocity[impact_index, 0]))
        after_wall = wall_indices[-1] + 1
        if after_wall <= number_of_steps:
            rebound_speed = float(center_velocity[after_wall, 0])
            if impact_speed > 0.0:
                coefficient_of_restitution = -rebound_speed / impact_speed
        wall_contact_duration = float((last_wall - first_wall + 1) * dt)
        diameter_before_impact = diameter[impact_index, 0]
        minimum_wall_diameter = float(diameter[wall_indices, 0].min())
        wall_compression = float(
            max(0.0, diameter_before_impact - minimum_wall_diameter)
        )
        # Use the maximum recovered diameter during the first 40 ms after
        # contact.  A fixed-time sample can be misleading because a compliant
        # ball oscillates around its original diameter.
        recovery_start = min(number_of_steps, last_wall + 1)
        recovery_end = min(
            number_of_steps, last_wall + max(1, round(0.040 / dt))
        )
        recovery_slice = slice(recovery_start, recovery_end + 1)
        recovery_offset = int(np.argmax(diameter[recovery_slice, 0]))
        recovery_index = recovery_start + recovery_offset
        wall_recovery_time = float(time[recovery_index] - time[last_wall])
        residual_index = min(
            number_of_steps, last_wall + max(1, round(0.040 / dt))
        )
        residual_radial_dent = float(radial_dent[residual_index])
        if wall_compression > 1.0e-12:
            wall_diameter_recovery = float(
                100.0
                * (diameter[recovery_index, 0] - minimum_wall_diameter)
                / wall_compression
            )

    if verbose:
        print("\nBaseball simulation summary")
        print(f"  radius / mass              : {radius:.6f} m / {mass:.4f} kg")
        print(f"  wall distance              : {wall_x:.2f} m")
        print(
            f"  surface/core nodes         : {surface_node_count} / 1"
        )
        print(f"  triangles / core springs   : {len(elements)} / {len(core_edges)}")
        print(f"  release speed              : {release_speed:.3f} m/s")
        print(
            f"  material rotation at release: "
            f"{twist_angle[release_index] / (2.0 * np.pi):.3f} turns"
        )
        print(f"  first wall contact         : {wall_hit_time} s")
        print(f"  first ground contact       : {ground_hit_time} s")
        print(f"  wall impact speed          : {impact_speed} m/s")
        print(f"  rebound x-velocity         : {rebound_speed} m/s")
        print(f"  wall coefficient restitution: {coefficient_of_restitution}")
        print(f"  wall contact duration      : {wall_contact_duration} s")
        print(f"  shell compression at wall : {wall_compression} m")
        if wall_recovery_time is None:
            print("  peak recovery within 40 ms: not reached")
        else:
            print(
                "  peak recovery within 40 ms: "
                f"{wall_diameter_recovery}% at "
                f"{1.0e3 * wall_recovery_time} ms"
            )
        print(f"  residual local dent at 40 ms: {residual_radial_dent} m")
        print(f"  peak pitcher force         : {pitcher_force_history.max():.3f} N")
        print(f"  peak pitcher torque        : {spin_torque_history.max():.6f} N m")
        print(f"  peak aerodynamic drag      : {drag_force_history.max():.3f} N")
        print(f"  peak wall force            : {wall_force.max():.3f} N")
        print(f"  peak ground force          : {ground_force.max():.3f} N")
        print(f"  max wall penetration       : {wall_penetration.max():.6e} m")
        print(f"  max ground penetration     : {ground_penetration.max():.6e} m")
        print(f"  peak core spring force     : {core_force_history.max():.3f} N")
        print(f"  maximum Newton iterations  : {maximum_newton_iterations}")

    return BaseballResult(
        time=time,
        states=states,
        stored_steps=np.asarray(stored_steps),
        center_of_mass=center_of_mass,
        center_velocity=center_velocity,
        diameter=diameter,
        radial_dent=radial_dent,
        pitcher_force=pitcher_force_history,
        spin_torque=spin_torque_history,
        twist_angle=twist_angle,
        drag_force=drag_force_history,
        wall_force=wall_force,
        ground_force=ground_force,
        wall_penetration=wall_penetration,
        ground_penetration=ground_penetration,
        stretching_force=stretching_force_history,
        bending_force=bending_force_history,
        core_force=core_force_history,
        elements=elements,
        wall_x=wall_x,
        radius=radius,
        mass=mass,
        throw_duration=throw_duration,
        wall_hit_time=wall_hit_time,
        ground_hit_time=ground_hit_time,
        release_speed=release_speed,
        rebound_speed=rebound_speed,
        impact_speed=impact_speed,
        coefficient_of_restitution=coefficient_of_restitution,
        wall_contact_duration=wall_contact_duration,
        wall_compression=wall_compression,
        wall_diameter_recovery=wall_diameter_recovery,
        wall_recovery_time=wall_recovery_time,
        residual_radial_dent=residual_radial_dent,
        marker_node=marker_node,
    )


def _stored_state(result: BaseballResult, step: int) -> np.ndarray:
    index = int(np.argmin(np.abs(result.stored_steps - step)))
    return result.states[index]


def plot_baseball_results(result: BaseballResult) -> None:
    wall_peak = int(np.argmax(result.wall_force))
    ground_peak = int(np.argmax(result.ground_force))
    throw_peak = int(np.argmax(result.pitcher_force))
    selected = [throw_peak, wall_peak, ground_peak, len(result.time) - 1]
    titles = ["Pitcher force", "Wall impact", "Ground impact", "Final"]

    shapes = plt.figure(figsize=(11, 9))
    view = 3.2 * result.radius
    for panel, (step, title) in enumerate(zip(selected, titles), start=1):
        xyz = _stored_state(result, step).reshape(-1, 3)
        center = result.center_of_mass[step]
        ax = shapes.add_subplot(2, 2, panel, projection="3d")
        ax.plot_trisurf(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=result.elements,
            color="cornflowerblue", edgecolor="black", linewidth=0.35, alpha=0.9,
        )
        ax.scatter(
            *xyz[result.marker_node], color="crimson", s=35, depthshade=False
        )
        ax.set_xlim(center[0] - view, center[0] + view)
        ax.set_ylim(center[1] - view, center[1] + view)
        ax.set_zlim(center[2] - view, center[2] + view)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{title}: t={result.time[step]:.3f} s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    shapes.suptitle("Soft robotic ball deformation (tracking views)")
    shapes.tight_layout()

    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes[0, 0].plot(result.center_of_mass[:, 0], result.center_of_mass[:, 2])
    axes[0, 0].axvline(result.wall_x, color="black", linestyle="--")
    axes[0, 0].axhline(0.0, color="gray", linestyle="--")
    axes[0, 0].set_xlabel("COM x [m]")
    axes[0, 0].set_ylabel("COM z [m]")
    axes[0, 0].set_title("Full pitch trajectory")

    speed = np.linalg.norm(result.center_velocity, axis=1)
    axes[0, 1].plot(result.time, speed)
    axes[0, 1].axvline(result.throw_duration, color="gray", linestyle="--")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("COM speed [m/s]")
    axes[0, 1].set_title("Acceleration and free flight")

    axes[0, 2].plot(result.time, result.twist_angle / (2.0 * np.pi))
    axes[0, 2].axvline(result.throw_duration, color="gray", linestyle="--")
    axes[0, 2].set_xlabel("time [s]")
    axes[0, 2].set_ylabel("material rotation [turns]")
    axes[0, 2].set_title("Backspin / shell twist")

    axes[0, 3].plot(result.time, result.pitcher_force, label="pitcher")
    axes[0, 3].plot(result.time, result.drag_force, label="air drag")
    axes[0, 3].set_xlabel("time [s]")
    axes[0, 3].set_ylabel("force [N]")
    axes[0, 3].set_title("External forces")
    axes[0, 3].legend()

    axes[1, 0].plot(result.time, result.wall_force, label="wall")
    axes[1, 0].plot(result.time, result.ground_force, label="ground")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("contact force [N]")
    axes[1, 0].set_title("Contact forces")
    axes[1, 0].legend()

    axes[1, 1].plot(result.time, result.diameter[:, 0], label="$D_x$")
    axes[1, 1].plot(result.time, result.diameter[:, 1], label="$D_y$")
    axes[1, 1].plot(result.time, result.diameter[:, 2], label="$D_z$")
    axes[1, 1].plot(
        result.time, result.radial_dent,
        color="crimson", linestyle="--", label="local dent",
    )
    axes[1, 1].axhline(2.0 * result.radius, color="black", linestyle="--")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("diameter [m]")
    axes[1, 1].set_title("Ball deformation")
    axes[1, 1].legend()

    axes[1, 2].plot(result.time, result.stretching_force, label="$F_s$")
    axes[1, 2].plot(result.time, result.bending_force, label="$F_b$")
    axes[1, 2].plot(result.time, result.core_force, label="$F_{core}$")
    axes[1, 2].set_xlabel("time [s]")
    axes[1, 2].set_ylabel("maximum nodal force [N]")
    axes[1, 2].set_title("Elastic response")
    axes[1, 2].legend()

    axes[1, 3].plot(result.time, result.spin_torque, color="darkorange")
    axes[1, 3].set_xlabel("time [s]")
    axes[1, 3].set_ylabel("torque [N m]")
    axes[1, 3].set_title("Pitcher spin torque")
    figure.tight_layout()


def animate_baseball(result: BaseballResult) -> FuncAnimation:
    figure = plt.figure(figsize=(13, 6))
    close_ax = figure.add_subplot(1, 2, 1, projection="3d")
    overview_ax = figure.add_subplot(1, 2, 2)
    xyz0 = result.states[0].reshape(-1, 3)
    shell = Poly3DCollection(
        xyz0[result.elements], facecolor="mediumpurple", edgecolor="black",
        linewidth=0.35, alpha=0.92,
    )
    close_ax.add_collection3d(shell)
    marker_xyz = xyz0[result.marker_node]
    marker = close_ax.scatter(
        [marker_xyz[0]], [marker_xyz[1]], [marker_xyz[2]],
        color="crimson", s=50, depthshade=False,
    )
    ground_x, ground_y = np.meshgrid(
        np.linspace(-0.2, result.wall_x, 2),
        np.linspace(-4.0 * result.radius, 4.0 * result.radius, 2),
    )
    close_ax.plot_surface(
        ground_x, ground_y, np.zeros_like(ground_x),
        color="lightgray", alpha=0.35,
    )
    wall_y, wall_z = np.meshgrid(
        np.linspace(-4.0 * result.radius, 4.0 * result.radius, 2),
        np.linspace(0.0, 2.1, 2),
    )
    close_ax.plot_surface(
        np.full_like(wall_y, result.wall_x), wall_y, wall_z,
        color="silver", alpha=0.30,
    )
    close_ax.set_box_aspect((1, 1, 1))
    close_ax.set_xlabel("x")
    close_ax.set_ylabel("y")
    close_ax.set_zlabel("z")
    close_ax.set_title("Tracking camera: shell deformation")
    time_label = close_ax.text2D(0.03, 0.95, "", transform=close_ax.transAxes)

    overview_ax.plot(
        result.center_of_mass[:, 0], result.center_of_mass[:, 2],
        color="lightgray", label="complete trajectory",
    )
    overview_ax.axvline(result.wall_x, color="black", linestyle="--", label="wall")
    overview_ax.axhline(0.0, color="gray", linestyle="--", label="ground")
    (completed_path,) = overview_ax.plot([], [], color="crimson", linewidth=2)
    (current_point,) = overview_ax.plot([], [], "o", color="royalblue")
    overview_ax.set_xlim(-0.2, result.wall_x + 0.3)
    overview_ax.set_ylim(-0.1, max(2.0, result.center_of_mass[:, 2].max() + 0.1))
    overview_ax.set_xlabel("x [m]")
    overview_ax.set_ylabel("z [m]")
    overview_ax.set_title("Pitch overview")
    overview_ax.legend()
    view = 3.2 * result.radius

    def update(frame: int):
        step = int(result.stored_steps[frame])
        xyz = result.states[frame].reshape(-1, 3)
        center = result.center_of_mass[step]
        shell.set_verts(xyz[result.elements])
        marker_xyz = xyz[result.marker_node]
        marker._offsets3d = (
            [marker_xyz[0]], [marker_xyz[1]], [marker_xyz[2]]
        )
        if result.pitcher_force[step] > 0.0:
            shell.set_facecolor("mediumpurple")
        elif result.wall_force[step] > 0.0:
            shell.set_facecolor("tomato")
        elif result.ground_force[step] > 0.0:
            shell.set_facecolor("gold")
        else:
            shell.set_facecolor("cornflowerblue")
        close_ax.set_xlim(center[0] - view, center[0] + view)
        close_ax.set_ylim(center[1] - view, center[1] + view)
        close_ax.set_zlim(center[2] - view, center[2] + view)
        time_label.set_text(
            f"t={result.time[step]:.3f} s, "
            f"speed={np.linalg.norm(result.center_velocity[step]):.1f} m/s, "
            f"rotation={result.twist_angle[step] / (2.0 * np.pi):.1f} turns"
        )

        path = result.center_of_mass[: step + 1]
        completed_path.set_data(path[:, 0], path[:, 2])
        current_point.set_data([center[0]], [center[2]])
        return shell, marker, completed_path, current_point, time_label

    return FuncAnimation(
        figure, update, frames=len(result.states), interval=35, repeat=True, blit=False
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=1, choices=(0, 1))
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--total-time", type=float, default=1.15)
    parser.add_argument("--output-stride", type=int, default=4)
    parser.add_argument("--youngs-modulus", type=float, default=8.0e7)
    parser.add_argument("--bending-multiplier", type=float, default=6.0)
    parser.add_argument("--core-stiffness", type=float, default=1.0e5)
    parser.add_argument("--spin-rpm", type=float, default=1200.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e6)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = run_baseball(
        subdivisions=args.subdivisions,
        dt=args.dt,
        total_time=args.total_time,
        output_stride=args.output_stride,
        effective_youngs_modulus=args.youngs_modulus,
        bending_multiplier=args.bending_multiplier,
        core_total_stiffness=args.core_stiffness,
        target_spin_rpm=args.spin_rpm,
        contact_stiffness=args.contact_stiffness,
        verbose=not args.quiet,
    )
    if not args.no_show:
        plot_baseball_results(result)
        animation = animate_baseball(result) if args.animate else None
        _ = animation
        plt.show()


if __name__ == "__main__":
    main()
