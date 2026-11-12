"""A deformable spherical shell thrown at a wall and then onto the ground.

The shell mechanics use the course-provided discrete elastic shell helpers in
``../mae-263F/src/shells``.  This file adds the spherical mesh, lumped mass,
gravity, plane contact, time integration, and plotting used by the project.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


COURSE_SHELL_DIR = (
    Path(__file__).resolve().parents[2] / "mae-263F" / "src" / "shells"
)
if not COURSE_SHELL_DIR.exists():
    raise FileNotFoundError(
        "Course shell helpers were not found at "
        f"{COURSE_SHELL_DIR}. Keep mae-263F next to mae-263f-coursework."
    )
sys.path.insert(0, str(COURSE_SHELL_DIR))

from shell_functions import getEdgesAndHinges, getFb, getFs, getThetaBar  # noqa: E402


@dataclass
class SimulationResult:
    time: np.ndarray
    states: list[np.ndarray]
    stored_steps: np.ndarray
    center_of_mass: np.ndarray
    diameter: np.ndarray
    wall_force: np.ndarray
    ground_force: np.ndarray
    wall_penetration: np.ndarray
    ground_penetration: np.ndarray
    stretching_force: np.ndarray
    bending_force: np.ndarray
    wall_hit_time: float | None
    ground_hit_time: float | None
    nodes0: np.ndarray
    elements: np.ndarray
    total_mass: float
    wall_x: float


def _oriented_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient every triangle so that its normal points away from the origin."""
    faces = faces.copy()
    for k, face in enumerate(faces):
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, (a + b + c) / 3.0) < 0.0:
            faces[k, 1], faces[k, 2] = faces[k, 2], faces[k, 1]
    return faces


def make_icosphere(
    radius: float,
    center: np.ndarray,
    subdivisions: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an outward-oriented triangular sphere without extra packages."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=int,
    )

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        vertex_list = [v.copy() for v in vertices]

        def midpoint(i: int, j: int) -> int:
            key = tuple(sorted((int(i), int(j))))
            if key not in midpoint_cache:
                point = vertices[key[0]] + vertices[key[1]]
                point /= np.linalg.norm(point)
                midpoint_cache[key] = len(vertex_list)
                vertex_list.append(point)
            return midpoint_cache[key]

        refined = []
        for i, j, k in faces:
            ij = midpoint(i, j)
            jk = midpoint(j, k)
            ki = midpoint(k, i)
            refined.extend(
                [(i, ij, ki), (j, jk, ij), (k, ki, jk), (ij, jk, ki)]
            )
        vertices = np.asarray(vertex_list)
        faces = np.asarray(refined, dtype=int)

    faces = _oriented_faces(vertices, faces)
    return radius * vertices + center, faces


def triangle_lumped_masses(
    nodes: np.ndarray,
    elements: np.ndarray,
    density: float,
    thickness: float,
) -> np.ndarray:
    """Distribute each triangular shell element's mass equally to its nodes."""
    nodal_mass = np.zeros(len(nodes))
    for triangle in elements:
        a, b, c = nodes[triangle]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        nodal_mass[triangle] += density * thickness * area / 3.0
    return nodal_mass


def plane_contact(
    q: np.ndarray,
    q_old: np.ndarray,
    dt: float,
    wall_x: float,
    stiffness: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Return frictionless penalty forces and Jacobian for wall and ground."""
    ndof = len(q)
    force = np.zeros(ndof)
    jacobian = np.zeros((ndof, ndof))
    velocity = (q - q_old) / dt

    wall_total = 0.0
    ground_total = 0.0
    wall_max_penetration = 0.0
    ground_max_penetration = 0.0

    for node in range(ndof // 3):
        ix = 3 * node
        iz = ix + 2

        wall_penetration = q[ix] - wall_x
        if wall_penetration > 0.0:
            normal_speed = velocity[ix]
            contact_force = -stiffness * wall_penetration
            derivative = -stiffness
            if normal_speed > 0.0:
                contact_force -= damping * normal_speed
                derivative -= damping / dt
            force[ix] += contact_force
            jacobian[ix, ix] += derivative
            wall_total += -contact_force
            wall_max_penetration = max(wall_max_penetration, wall_penetration)

        ground_penetration = -q[iz]
        if ground_penetration > 0.0:
            normal_speed = velocity[iz]
            contact_force = stiffness * ground_penetration
            derivative = -stiffness
            if normal_speed < 0.0:
                contact_force -= damping * normal_speed
                derivative -= damping / dt
            force[iz] += contact_force
            jacobian[iz, iz] += derivative
            ground_total += contact_force
            ground_max_penetration = max(
                ground_max_penetration, ground_penetration
            )

    return (
        force,
        jacobian,
        wall_total,
        ground_total,
        wall_max_penetration,
        ground_max_penetration,
    )


def implicit_step(
    q_old: np.ndarray,
    velocity_old: np.ndarray,
    dt: float,
    mass_vector: np.ndarray,
    edges: np.ndarray,
    reference_lengths: np.ndarray,
    stretching_stiffness: np.ndarray,
    hinges: np.ndarray,
    natural_angles: np.ndarray,
    bending_stiffness: float,
    gravity_force: np.ndarray,
    wall_x: float,
    contact_stiffness: float,
    contact_damping: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], int]:
    """Advance one backward-Euler step using Newton's method."""
    q = q_old + dt * velocity_old
    mass_over_dt2 = mass_vector / dt**2

    for iteration in range(1, maximum_iterations + 1):
        bending_force, bending_jacobian = getFb(
            q, hinges, natural_angles, bending_stiffness
        )
        stretching_force, stretching_jacobian = getFs(
            q, edges, reference_lengths, stretching_stiffness
        )
        (
            contact_force,
            contact_jacobian,
            wall_force,
            ground_force,
            wall_penetration,
            ground_penetration,
        ) = plane_contact(
            q,
            q_old,
            dt,
            wall_x,
            contact_stiffness,
            contact_damping,
        )

        total_force = (
            bending_force + stretching_force + gravity_force + contact_force
        )
        force_jacobian = (
            bending_jacobian + stretching_jacobian + contact_jacobian
        )
        residual = mass_vector * (
            (q - q_old) / dt**2 - velocity_old / dt
        ) - total_force

        error = np.linalg.norm(residual, ord=np.inf)
        if error <= tolerance:
            break

        system_matrix = np.diag(mass_over_dt2) - force_jacobian
        update = np.linalg.solve(system_matrix, residual)
        q -= update
    else:
        raise RuntimeError(
            f"Newton did not converge after {maximum_iterations} iterations; "
            f"residual={error:.3e} N"
        )

    velocity = (q - q_old) / dt
    maximum_stretching_force = float(
        np.linalg.norm(stretching_force.reshape(-1, 3), axis=1).max()
    )
    maximum_bending_force = float(
        np.linalg.norm(bending_force.reshape(-1, 3), axis=1).max()
    )
    contact_summary = (
        wall_force,
        ground_force,
        wall_penetration,
        ground_penetration,
        maximum_stretching_force,
        maximum_bending_force,
    )
    return q, velocity, contact_summary, iteration


def run_simulation(
    *,
    subdivisions: int = 1,
    dt: float = 0.004,
    total_time: float = 1.0,
    output_stride: int = 2,
    verbose: bool = True,
) -> SimulationResult:
    """Run the wall-then-ground collision example."""
    radius = 0.10
    center0 = np.array([0.0, 0.0, 1.0])
    wall_x = 0.90
    youngs_modulus = 2.0e5
    thickness = 0.005
    density = 1000.0
    gravity = np.array([0.0, 0.0, -9.81])
    initial_velocity = np.array([2.5, 0.0, 0.2])

    contact_stiffness = 2.0e4
    # Start with conservative contact. A velocity-dependent dashpot can be added
    # later, but its switching condition makes the first Newton contact step less
    # robust and is not needed to demonstrate deformation and rebound.
    contact_damping = 0.0
    tolerance = 1.0e-5
    maximum_iterations = 30

    nodes0, elements = make_icosphere(radius, center0, subdivisions)
    edges, hinges = getEdgesAndHinges(elements)
    q = nodes0.reshape(-1).copy()
    velocity = np.tile(initial_velocity, len(nodes0))

    reference_lengths = np.linalg.norm(
        nodes0[edges[:, 1]] - nodes0[edges[:, 0]], axis=1
    )
    stretching_stiffness = (
        np.sqrt(3.0) / 2.0
        * youngs_modulus
        * thickness
        * reference_lengths
    )
    bending_stiffness = (
        2.0 / np.sqrt(3.0) * youngs_modulus * thickness**3 / 12.0
    )
    natural_angles = getThetaBar(q, hinges)

    nodal_mass = triangle_lumped_masses(
        nodes0, elements, density, thickness
    )
    mass_vector = np.repeat(nodal_mass, 3)
    gravity_force = mass_vector * np.tile(gravity, len(nodes0))

    number_of_steps = int(round(total_time / dt))
    time = np.arange(number_of_steps + 1) * dt
    center_of_mass = np.zeros((number_of_steps + 1, 3))
    diameter = np.zeros((number_of_steps + 1, 3))
    wall_force = np.zeros(number_of_steps + 1)
    ground_force = np.zeros(number_of_steps + 1)
    wall_penetration = np.zeros(number_of_steps + 1)
    ground_penetration = np.zeros(number_of_steps + 1)
    maximum_stretching_force = np.zeros(number_of_steps + 1)
    maximum_bending_force = np.zeros(number_of_steps + 1)
    stored_steps = [0]
    states = [q.copy()]

    def record(step: int) -> None:
        positions = q.reshape(-1, 3)
        center_of_mass[step] = np.average(
            positions, axis=0, weights=nodal_mass
        )
        diameter[step] = np.ptp(positions, axis=0)

    record(0)
    maximum_newton_iterations = 0

    for step in range(1, number_of_steps + 1):
        q, velocity, contact, iterations = implicit_step(
            q,
            velocity,
            dt,
            mass_vector,
            edges,
            reference_lengths,
            stretching_stiffness,
            hinges,
            natural_angles,
            bending_stiffness,
            gravity_force,
            wall_x,
            contact_stiffness,
            contact_damping,
            tolerance,
            maximum_iterations,
        )
        (
            wall_force[step],
            ground_force[step],
            wall_penetration[step],
            ground_penetration[step],
            maximum_stretching_force[step],
            maximum_bending_force[step],
        ) = contact
        record(step)
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)

        if step % output_stride == 0 or step == number_of_steps:
            stored_steps.append(step)
            states.append(q.copy())
        if verbose and step % max(1, number_of_steps // 10) == 0:
            print(
                f"t={time[step]:.3f} s, "
                f"COM=({center_of_mass[step, 0]:.3f}, "
                f"{center_of_mass[step, 2]:.3f}) m"
            )

    wall_indices = np.flatnonzero(wall_force > 1.0e-8)
    ground_indices = np.flatnonzero(ground_force > 1.0e-8)
    wall_hit_time = float(time[wall_indices[0]]) if wall_indices.size else None
    ground_hit_time = (
        float(time[ground_indices[0]]) if ground_indices.size else None
    )

    if verbose:
        print("\nSimulation summary")
        print(f"  nodes / triangles / hinges : {len(nodes0)} / {len(elements)} / {len(hinges)}")
        print(f"  total shell mass           : {nodal_mass.sum():.6f} kg")
        print(f"  first wall contact         : {wall_hit_time} s")
        print(f"  first ground contact       : {ground_hit_time} s")
        print(f"  peak wall force            : {wall_force.max():.6f} N")
        print(f"  peak ground force          : {ground_force.max():.6f} N")
        print(f"  max wall penetration       : {wall_penetration.max():.6e} m")
        print(f"  max ground penetration     : {ground_penetration.max():.6e} m")
        print(f"  peak stretching force      : {maximum_stretching_force.max():.6f} N")
        print(f"  peak bending force         : {maximum_bending_force.max():.6f} N")
        print(f"  maximum Newton iterations  : {maximum_newton_iterations}")

    return SimulationResult(
        time=time,
        states=states,
        stored_steps=np.asarray(stored_steps),
        center_of_mass=center_of_mass,
        diameter=diameter,
        wall_force=wall_force,
        ground_force=ground_force,
        wall_penetration=wall_penetration,
        ground_penetration=ground_penetration,
        stretching_force=maximum_stretching_force,
        bending_force=maximum_bending_force,
        wall_hit_time=wall_hit_time,
        ground_hit_time=ground_hit_time,
        nodes0=nodes0,
        elements=elements,
        total_mass=float(nodal_mass.sum()),
        wall_x=wall_x,
    )


def _closest_stored_state(result: SimulationResult, step: int) -> np.ndarray:
    index = int(np.argmin(np.abs(result.stored_steps - step)))
    return result.states[index]


def plot_results(result: SimulationResult) -> None:
    """Create temporary shape snapshots and diagnostic plots."""
    wall_peak = int(np.argmax(result.wall_force))
    ground_peak = int(np.argmax(result.ground_force))
    selected_steps = [0, wall_peak, ground_peak, len(result.time) - 1]
    selected_titles = ["Initial", "Wall impact", "Ground impact", "Final"]

    figure = plt.figure(figsize=(12, 9))
    for panel, (step, title) in enumerate(
        zip(selected_steps, selected_titles), start=1
    ):
        q = _closest_stored_state(result, step)
        xyz = q.reshape(-1, 3)
        ax = figure.add_subplot(2, 2, panel, projection="3d")
        ax.plot_trisurf(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            triangles=result.elements,
            color="cornflowerblue",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
        )
        ax.set_title(f"{title}: t={result.time[step]:.3f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_xlim(-0.25, result.wall_x + 0.08)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0.0, 1.15)
        ax.set_box_aspect((1.23, 0.70, 1.15))
    figure.suptitle("Deformable shell: wall collision followed by ground collision")
    figure.tight_layout()

    diagnostics, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].plot(result.center_of_mass[:, 0], result.center_of_mass[:, 2])
    axes[0, 0].axvline(result.wall_x, color="black", linestyle="--", label="wall")
    axes[0, 0].axhline(0.0, color="gray", linestyle="--", label="ground")
    axes[0, 0].set_xlabel("COM x [m]")
    axes[0, 0].set_ylabel("COM z [m]")
    axes[0, 0].set_title("Center-of-mass trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(result.time, result.wall_force, label="wall")
    axes[0, 1].plot(result.time, result.ground_force, label="ground")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("total contact force [N]")
    axes[0, 1].set_title("Contact forces")
    axes[0, 1].legend()

    axes[1, 0].plot(result.time, result.diameter[:, 0], label="$D_x$")
    axes[1, 0].plot(result.time, result.diameter[:, 1], label="$D_y$")
    axes[1, 0].plot(result.time, result.diameter[:, 2], label="$D_z$")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("shell diameter [m]")
    axes[1, 0].set_title("Deformation of the shell")
    axes[1, 0].legend()

    axes[1, 1].plot(result.time, result.wall_penetration, label="wall")
    axes[1, 1].plot(result.time, result.ground_penetration, label="ground")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("maximum penetration [m]")
    axes[1, 1].set_title("Penalty-contact penetration")
    axes[1, 1].legend()

    axes[1, 2].plot(
        result.time, result.stretching_force, label="stretching $F_s$"
    )
    axes[1, 2].plot(
        result.time, result.bending_force, label="bending $F_b$"
    )
    axes[1, 2].set_xlabel("time [s]")
    axes[1, 2].set_ylabel("maximum nodal force [N]")
    axes[1, 2].set_title("Elastic-force response")
    axes[1, 2].legend()

    diagnostics.tight_layout()


def animate_result(result: SimulationResult) -> FuncAnimation:
    """Create an interactive animation without saving an image or video file."""
    figure = plt.figure(figsize=(9, 7))
    ax = figure.add_subplot(111, projection="3d")
    xyz0 = result.states[0].reshape(-1, 3)
    shell = Poly3DCollection(
        xyz0[result.elements],
        facecolor="cornflowerblue",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.90,
    )
    ax.add_collection3d(shell)

    ground_x, ground_y = np.meshgrid(
        np.linspace(-0.25, result.wall_x, 2), np.linspace(-0.35, 0.35, 2)
    )
    ax.plot_surface(
        ground_x,
        ground_y,
        np.zeros_like(ground_x),
        color="lightgray",
        alpha=0.45,
    )
    wall_y, wall_z = np.meshgrid(
        np.linspace(-0.35, 0.35, 2), np.linspace(0.0, 1.15, 2)
    )
    ax.plot_surface(
        np.full_like(wall_y, result.wall_x),
        wall_y,
        wall_z,
        color="silver",
        alpha=0.35,
    )

    (trajectory,) = ax.plot([], [], [], color="crimson", linewidth=1.5)
    time_label = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    force_label = ax.text2D(0.03, 0.90, "", transform=ax.transAxes)
    ax.set_xlim(-0.25, result.wall_x + 0.08)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(0.0, 1.15)
    ax.set_box_aspect((1.23, 0.70, 1.15))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Deformable spherical shell collision")

    def update(frame: int):
        step = int(result.stored_steps[frame])
        xyz = result.states[frame].reshape(-1, 3)
        shell.set_verts(xyz[result.elements])
        if result.wall_force[step] > 0.0:
            shell.set_facecolor("tomato")
        elif result.ground_force[step] > 0.0:
            shell.set_facecolor("gold")
        else:
            shell.set_facecolor("cornflowerblue")

        path = result.center_of_mass[result.stored_steps[: frame + 1]]
        trajectory.set_data(path[:, 0], path[:, 1])
        trajectory.set_3d_properties(path[:, 2])
        time_label.set_text(f"t = {result.time[step]:.3f} s")
        force_label.set_text(
            f"max $F_s$ = {result.stretching_force[step]:.2f} N, "
            f"max $F_b$ = {result.bending_force[step]:.2f} N"
        )
        return shell, trajectory, time_label, force_label

    return FuncAnimation(
        figure,
        update,
        frames=len(result.states),
        interval=40,
        repeat=True,
        blit=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=1, choices=(0, 1))
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--animate", action="store_true", help="also display a live 3D animation"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = run_simulation(
        subdivisions=args.subdivisions,
        dt=args.dt,
        total_time=args.total_time,
        output_stride=args.output_stride,
        verbose=not args.quiet,
    )
    if not args.no_show:
        plot_results(result)
        animation = animate_result(result) if args.animate else None
        # Keep the animation referenced until the interactive window closes.
        _ = animation
        plt.show()


if __name__ == "__main__":
    main()
