"""Predict a rigid arrow launched by a deformable bowstring.

The bow endpoints are fixed, the string is a discrete stretching-only soft body,
and the arrow is a rigid translating body.  After the string reaches the brace
plane, the arrow detaches and follows a gravity-plus-quadratic-drag trajectory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

HOMEWORK_2_DIR = Path(__file__).resolve().parents[1] / "homework_2"
sys.path.insert(0, str(HOMEWORK_2_DIR))
from elastic_derivatives import stretching_gradient_hessian  # noqa: E402


@dataclass
class ArcheryResult:
    time: np.ndarray
    string_states: list[np.ndarray]
    string_velocity: np.ndarray
    nock_position: np.ndarray
    arrow_speed: np.ndarray
    nock_force: np.ndarray
    string_energy: np.ndarray
    string_kinetic_energy: np.ndarray
    arrow_kinetic_energy: np.ndarray
    release_time: float
    release_speed: float
    initial_string_energy: float
    energy_limit_speed: float
    effective_EA: float
    draw_force: float
    required_aim_angle: float
    flight_time: np.ndarray
    flight_position: np.ndarray
    flight_velocity: np.ndarray
    target_distance: float
    target_height: float
    target_time: float
    target_speed: float
    target_height_error: float
    arrow_mass: float
    arrow_length: float
    half_span: float
    draw_length: float
    midpoint: int


def initial_string_nodes(
    node_count: int, half_span: float, draw_length: float
) -> np.ndarray:
    """Return a symmetric V-shaped string with its nock pulled backward."""
    if node_count < 5 or node_count % 2 == 0:
        raise ValueError("node_count must be an odd integer of at least 5")
    y = np.linspace(-half_span, half_span, node_count)
    x = -draw_length * (1.0 - np.abs(y) / half_span)
    return np.column_stack((x, y))


def effective_axial_stiffness(
    draw_force: float, half_span: float, draw_length: float
) -> float:
    """Calibrate EA so the initial V-shaped string has the requested draw force."""
    half_length = np.hypot(half_span, draw_length)
    strain = (half_length - half_span) / half_span
    horizontal_projection = draw_length / half_length
    return draw_force / (2.0 * strain * horizontal_projection)


def assemble_string_elasticity(
    q: np.ndarray, rest_length: float, EA: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Assemble stretching energy, gradient, and Hessian for all string edges."""
    node_count = len(q) // 2
    energy = 0.0
    gradient = np.zeros_like(q)
    hessian = np.zeros((len(q), len(q)))
    for edge in range(node_count - 1):
        indices = np.array(
            [2 * edge, 2 * edge + 1, 2 * edge + 2, 2 * edge + 3]
        )
        local_q = q[indices]
        length = np.linalg.norm(local_q[2:4] - local_q[0:2])
        energy += 0.5 * EA / rest_length * (length - rest_length) ** 2
        local_gradient, local_hessian = stretching_gradient_hessian(
            local_q, rest_length, EA
        )
        gradient[indices] += local_gradient
        hessian[np.ix_(indices, indices)] += local_hessian
    return energy, gradient, hessian


def implicit_string_step(
    q_old: np.ndarray,
    velocity_old: np.ndarray,
    dt: float,
    mass_vector: np.ndarray,
    free_indices: np.ndarray,
    rest_length: float,
    EA: float,
    damping: float,
    tolerance: float = 1.0e-7,
    maximum_iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One backward-Euler/Newton step for the deformable string."""
    q = q_old + dt * velocity_old
    for iteration in range(1, maximum_iterations + 1):
        _, elastic_gradient, elastic_hessian = assemble_string_elasticity(
            q, rest_length, EA
        )
        velocity = (q - q_old) / dt
        elastic_force = -elastic_gradient
        damping_force = -damping * velocity
        total_force = elastic_force + damping_force
        force_jacobian = -elastic_hessian - damping / dt * np.eye(len(q))
        residual = mass_vector * (
            (q - q_old) / dt**2 - velocity_old / dt
        ) - total_force
        if np.linalg.norm(residual[free_indices], ord=np.inf) <= tolerance:
            break
        system = np.diag(mass_vector / dt**2) - force_jacobian
        increment = np.linalg.solve(
            system[np.ix_(free_indices, free_indices)],
            residual[free_indices],
        )
        q[free_indices] -= increment
    else:
        raise RuntimeError(
            f"string Newton solve did not converge; residual="
            f"{np.linalg.norm(residual[free_indices], ord=np.inf):.3e}"
        )
    return q, (q - q_old) / dt, iteration


def arrow_acceleration(
    velocity: np.ndarray,
    mass: float,
    diameter: float,
    air_density: float,
    drag_coefficient: float,
) -> np.ndarray:
    speed = np.linalg.norm(velocity)
    area = np.pi * diameter**2 / 4.0
    drag = (
        -0.5 * air_density * drag_coefficient * area * speed * velocity
        if speed > 0.0
        else np.zeros(2)
    )
    return drag / mass + np.array([0.0, -9.81])


def simulate_flight(
    release_speed: float,
    aim_angle_degrees: float,
    target_distance: float,
    initial_height: float,
    arrow_mass: float,
    arrow_diameter: float,
    flight_dt: float = 0.001,
    maximum_time: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the rigid arrow trajectory with RK4."""
    angle = np.deg2rad(aim_angle_degrees)
    position = np.array([0.0, initial_height])
    velocity = release_speed * np.array([np.cos(angle), np.sin(angle)])
    times = [0.0]
    positions = [position.copy()]
    velocities = [velocity.copy()]

    def derivative(state: np.ndarray) -> np.ndarray:
        v = state[2:4]
        return np.r_[
            v,
            arrow_acceleration(v, arrow_mass, arrow_diameter, 1.225, 1.0),
        ]

    steps = int(np.ceil(maximum_time / flight_dt))
    state = np.r_[position, velocity]
    for step in range(1, steps + 1):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * flight_dt * k1)
        k3 = derivative(state + 0.5 * flight_dt * k2)
        k4 = derivative(state + flight_dt * k3)
        state = state + flight_dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        times.append(step * flight_dt)
        positions.append(state[:2].copy())
        velocities.append(state[2:4].copy())
        if state[0] >= target_distance:
            break
    return np.asarray(times), np.asarray(positions), np.asarray(velocities)


def interpolate_at_target(
    positions: np.ndarray,
    values: np.ndarray,
    target_distance: float,
) -> np.ndarray:
    index = int(np.searchsorted(positions[:, 0], target_distance))
    if index == 0 or index >= len(positions):
        raise RuntimeError("arrow did not reach the target distance")
    x0, x1 = positions[index - 1 : index + 1, 0]
    fraction = (target_distance - x0) / (x1 - x0)
    return values[index - 1] + fraction * (values[index] - values[index - 1])


def required_aim_angle(
    release_speed: float,
    target_distance: float,
    initial_height: float,
    target_height: float,
    arrow_mass: float,
    arrow_diameter: float,
) -> float:
    """Find the low-angle trajectory that reaches the target height."""
    lower = 0.0
    upper = 30.0

    def height_error(angle: float) -> float:
        _, positions, _ = simulate_flight(
            release_speed,
            angle,
            target_distance,
            initial_height,
            arrow_mass,
            arrow_diameter,
        )
        height = float(
            interpolate_at_target(positions, positions[:, 1], target_distance)
        )
        return height - target_height

    if height_error(lower) * height_error(upper) > 0.0:
        raise RuntimeError("could not bracket an aim angle for this target")
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        if height_error(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def run_archery(
    *,
    node_count: int = 21,
    launch_dt: float = 1.0e-4,
    launch_duration: float = 0.040,
    half_span: float = 0.60,
    draw_length: float = 0.70,
    draw_force: float = 200.0,
    string_mass: float = 0.006,
    arrow_mass: float = 0.025,
    arrow_length: float = 0.75,
    arrow_diameter: float = 0.006,
    damping: float = 0.002,
    target_distance: float = 70.0,
    initial_height: float = 1.50,
    target_height: float = 1.50,
    verbose: bool = True,
) -> ArcheryResult:
    nodes0 = initial_string_nodes(node_count, half_span, draw_length)
    q = nodes0.reshape(-1).copy()
    velocity = np.zeros_like(q)
    midpoint = node_count // 2
    rest_length = 2.0 * half_span / (node_count - 1)
    EA = effective_axial_stiffness(draw_force, half_span, draw_length)

    nodal_mass = np.full(node_count, string_mass / (node_count - 1))
    nodal_mass[0] *= 0.5
    nodal_mass[-1] *= 0.5
    mass_vector_string = np.repeat(nodal_mass, 2)
    fixed_indices = np.array([0, 1, 2 * (node_count - 1), 2 * node_count - 1])
    free_indices = np.setdiff1d(np.arange(2 * node_count), fixed_indices)

    step_count = int(round(launch_duration / launch_dt))
    time = np.arange(step_count + 1) * launch_dt
    states = [q.copy()]
    string_velocity = np.zeros((step_count + 1, 2 * node_count))
    nock_position = np.zeros(step_count + 1)
    arrow_speed = np.zeros(step_count + 1)
    nock_force = np.zeros(step_count + 1)
    string_energy = np.zeros(step_count + 1)
    string_kinetic = np.zeros(step_count + 1)
    arrow_kinetic = np.zeros(step_count + 1)
    attached = True
    release_time = None
    release_speed = None
    maximum_iterations = 0

    initial_energy, _, _ = assemble_string_elasticity(q, rest_length, EA)
    for step in range(step_count + 1):
        energy, gradient, _ = assemble_string_elasticity(q, rest_length, EA)
        string_energy[step] = energy
        string_kinetic[step] = 0.5 * np.sum(
            mass_vector_string * velocity**2
        )
        nock_position[step] = q[2 * midpoint]
        if attached:
            arrow_speed[step] = velocity[2 * midpoint]
        else:
            arrow_speed[step] = float(release_speed)
        arrow_kinetic[step] = 0.5 * arrow_mass * arrow_speed[step] ** 2
        nock_force[step] = max(0.0, -gradient[2 * midpoint])
        string_velocity[step] = velocity
        if step == step_count:
            break

        mass_vector = mass_vector_string.copy()
        if attached:
            mass_vector[2 * midpoint] += arrow_mass
        q_old = q.copy()
        q, velocity, iterations = implicit_string_step(
            q,
            velocity,
            launch_dt,
            mass_vector,
            free_indices,
            rest_length,
            EA,
            damping,
        )
        maximum_iterations = max(maximum_iterations, iterations)
        if attached and q[2 * midpoint] >= 0.0:
            attached = False
            release_time = time[step + 1]
            release_speed = max(0.0, velocity[2 * midpoint])
            q[2 * midpoint] = 0.0
            # Preserve the string-node velocity after the arrow separates.
            if q_old[2 * midpoint] < 0.0:
                nock_position[step + 1] = 0.0
        states.append(q.copy())

    if release_time is None or release_speed is None:
        raise RuntimeError("arrow did not detach during launch_duration")

    energy_limit_speed = np.sqrt(2.0 * initial_energy / arrow_mass)
    aim_angle = required_aim_angle(
        release_speed,
        target_distance,
        initial_height,
        target_height,
        arrow_mass,
        arrow_diameter,
    )
    flight_time, flight_position, flight_velocity = simulate_flight(
        release_speed,
        aim_angle,
        target_distance,
        initial_height,
        arrow_mass,
        arrow_diameter,
    )
    target_z = float(
        interpolate_at_target(flight_position, flight_position[:, 1], target_distance)
    )
    target_t = float(
        interpolate_at_target(flight_position, flight_time, target_distance)
    )
    target_v = interpolate_at_target(
        flight_position, flight_velocity, target_distance
    )

    if verbose:
        print("Archery prediction")
        print(f"  string nodes / edges       : {node_count} / {node_count - 1}")
        print(f"  calibrated EA              : {EA:.3f} N")
        print(f"  initial string energy      : {initial_energy:.3f} J")
        print(f"  arrow release time         : {release_time * 1e3:.3f} ms")
        print(f"  predicted release speed    : {release_speed:.3f} m/s")
        print(f"  energy-limit speed         : {energy_limit_speed:.3f} m/s")
        print(
            f"  launch energy efficiency   : "
            f"{100.0 * 0.5 * arrow_mass * release_speed**2 / initial_energy:.2f}%"
        )
        print(f"  required aim angle         : {aim_angle:.3f} deg")
        print(f"  time to {target_distance:g} m target      : {target_t:.3f} s")
        print(f"  speed at target            : {np.linalg.norm(target_v):.3f} m/s")
        print(f"  target height error        : {target_z - target_height:.6f} m")
        print(f"  maximum Newton iterations  : {maximum_iterations}")

    return ArcheryResult(
        time=time,
        string_states=states,
        string_velocity=string_velocity,
        nock_position=nock_position,
        arrow_speed=arrow_speed,
        nock_force=nock_force,
        string_energy=string_energy,
        string_kinetic_energy=string_kinetic,
        arrow_kinetic_energy=arrow_kinetic,
        release_time=float(release_time),
        release_speed=float(release_speed),
        initial_string_energy=float(initial_energy),
        energy_limit_speed=float(energy_limit_speed),
        effective_EA=float(EA),
        draw_force=draw_force,
        required_aim_angle=float(aim_angle),
        flight_time=flight_time,
        flight_position=flight_position,
        flight_velocity=flight_velocity,
        target_distance=target_distance,
        target_height=target_height,
        target_time=target_t,
        target_speed=float(np.linalg.norm(target_v)),
        target_height_error=target_z - target_height,
        arrow_mass=arrow_mass,
        arrow_length=arrow_length,
        half_span=half_span,
        draw_length=draw_length,
        midpoint=midpoint,
    )


def plot_archery_results(result: ArcheryResult) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    release_index = int(np.searchsorted(result.time, result.release_time))
    snapshot_indices = [0, max(1, release_index // 2), release_index, -1]
    for index, label in zip(
        snapshot_indices, ["drawn", "accelerating", "release", "string vibration"]
    ):
        nodes = result.string_states[index].reshape(-1, 2)
        axes[0, 0].plot(nodes[:, 0], nodes[:, 1], label=label)
    axes[0, 0].plot(
        [0.0, 0.0], [-result.half_span, result.half_span],
        color="saddlebrown", linewidth=5, label="rigid bow",
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("arrow direction x [m]")
    axes[0, 0].set_ylabel("bow span y [m]")
    axes[0, 0].set_title("Soft-string launch sequence")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(result.time * 1e3, result.arrow_speed, label="arrow speed")
    axes[0, 1].set_xlabel("time [ms]")
    axes[0, 1].set_ylabel("speed [m/s]")
    force_axis = axes[0, 1].twinx()
    force_axis.plot(
        result.time * 1e3, result.nock_force,
        color="darkorange", alpha=0.75, label="nock force",
    )
    force_axis.set_ylabel("string force [N]")
    axes[0, 1].axvline(result.release_time * 1e3, color="black", linestyle="--")
    axes[0, 1].set_title("Rigid-arrow acceleration")
    speed_lines, speed_labels = axes[0, 1].get_legend_handles_labels()
    force_lines, force_labels = force_axis.get_legend_handles_labels()
    axes[0, 1].legend(
        speed_lines + force_lines,
        speed_labels + force_labels,
        loc="center right",
    )

    axes[1, 0].plot(result.time * 1e3, result.string_energy, label="string elastic")
    axes[1, 0].plot(
        result.time * 1e3, result.string_kinetic_energy, label="string kinetic"
    )
    axes[1, 0].plot(
        result.time * 1e3, result.arrow_kinetic_energy, label="arrow kinetic"
    )
    axes[1, 0].axhline(
        result.initial_string_energy, color="gray", linestyle="--", label="initial"
    )
    axes[1, 0].set_xlabel("time [ms]")
    axes[1, 0].set_ylabel("energy [J]")
    axes[1, 0].set_title("Energy transfer")
    axes[1, 0].legend()

    axes[1, 1].plot(
        result.flight_position[:, 0], result.flight_position[:, 1], label="prediction"
    )
    axes[1, 1].plot(
        [result.target_distance], [result.target_height],
        marker="o", markersize=8, color="crimson", label="target",
    )
    axes[1, 1].axhline(0.0, color="gray", linestyle="--")
    axes[1, 1].set_xlabel("horizontal distance [m]")
    axes[1, 1].set_ylabel("height [m]")
    axes[1, 1].set_title(
        f"Trajectory: required aim = {result.required_aim_angle:.2f} deg"
    )
    axes[1, 1].legend()
    figure.tight_layout()
    return figure


def animate_launch(result: ArcheryResult) -> FuncAnimation:
    figure, axis = plt.subplots(figsize=(10, 6))
    nodes0 = result.string_states[0].reshape(-1, 2)
    (string_line,) = axis.plot(nodes0[:, 0], nodes0[:, 1], "o-", label="soft string")
    axis.plot(
        [0.0, 0.0], [-result.half_span, result.half_span],
        color="saddlebrown", linewidth=7, label="rigid bow",
    )
    (arrow_line,) = axis.plot([], [], color="crimson", linewidth=3, label="rigid arrow")
    time_text = axis.text(0.03, 0.95, "", transform=axis.transAxes)
    arrow_travel = result.release_speed * (result.time[-1] - result.release_time)
    axis.set_xlim(
        -result.draw_length - 0.1,
        max(result.arrow_length + 0.35, arrow_travel + result.arrow_length + 0.1),
    )
    axis.set_ylim(-1.15 * result.half_span, 1.15 * result.half_span)
    axis.set_aspect("equal")
    axis.set_xlabel("arrow direction x [m]")
    axis.set_ylabel("bow span y [m]")
    axis.legend()

    stride = max(1, len(result.time) // 240)
    frames = np.arange(0, len(result.time), stride)

    def update(frame: int):
        nodes = result.string_states[frame].reshape(-1, 2)
        string_line.set_data(nodes[:, 0], nodes[:, 1])
        if result.time[frame] <= result.release_time:
            nock_x = nodes[result.midpoint, 0]
        else:
            nock_x = result.release_speed * (result.time[frame] - result.release_time)
        arrow_line.set_data(
            [nock_x, nock_x + result.arrow_length], [0.0, 0.0]
        )
        time_text.set_text(
            f"t={1e3 * result.time[frame]:.2f} ms, "
            f"arrow speed={result.arrow_speed[frame]:.1f} m/s"
        )
        return string_line, arrow_line, time_text

    return FuncAnimation(
        figure, update, frames=frames, interval=30, repeat=True, blit=False
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=21)
    parser.add_argument("--draw-force", type=float, default=200.0)
    parser.add_argument("--draw-length", type=float, default=0.70)
    parser.add_argument("--target-distance", type=float, default=70.0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = run_archery(
        node_count=args.nodes,
        draw_force=args.draw_force,
        draw_length=args.draw_length,
        target_distance=args.target_distance,
        verbose=not args.quiet,
    )
    if not args.no_show:
        plot_archery_results(result)
        animation = animate_launch(result) if args.animate else None
        _ = animation
        plt.show()


if __name__ == "__main__":
    main()
