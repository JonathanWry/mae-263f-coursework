"""Homework 3: three-dimensional Discrete Elastic Rod simulations.

The DER force and Jacobian routines are the course-provided implementations in
``homework_3/src/der`` and ``homework_3/appendix``.  This file supplies the
problem setup, implicit-Euler/Newton solver, parameter sweeps, and question
outputs required by the homework handout.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "der"))

from der_functions import (  # noqa: E402
    assignOrthonormalAdaptedFrame,
    computeMaterialDirectors,
    computeSpaceParallel,
    computeTangent,
    computeTimeParallel,
    getFb,
    getFs,
    getFt,
    getKappa,
    getRefTwist,
)


@dataclass
class RodResult:
    """Final state and compact histories from one DER simulation."""

    q: np.ndarray
    time: np.ndarray
    tip_z: np.ndarray
    end_distance: np.ndarray
    reference_lengths: np.ndarray
    final_twist: np.ndarray
    steady_relative_change: float
    maximum_newton_iterations: int
    final_velocity_norm: float
    free_static_residual_norm: float
    relative_static_residual: float


def node_positions(q):
    """Return the ``(N, 3)`` node-position array stored in a DER DOF vector."""

    return np.column_stack((q[0::4], q[1::4], q[2::4]))


def euler_bernoulli_deflection(
    youngs_modulus,
    gravity,
    length=0.2,
    radius=1.0e-3,
    density=1000.0,
):
    """Cantilever tip deflection under its uniformly distributed self-weight."""

    area = np.pi * radius**2
    second_moment = np.pi * radius**4 / 4.0
    weight_per_length = density * area * gravity
    return weight_per_length * length**4 / (
        8.0 * youngs_modulus * second_moment
    )


def _initial_geometry(nv, length, natural_radius):
    """Build the straight Part 1 or naturally curved Part 2 centerline."""

    nodes = np.zeros((nv, 3))
    if natural_radius == 0.0:
        # The first edge lies inside the clamp.  The remaining N-2 edges have
        # total free length L, so node 1 is at x=0 and the tip is at x=L.
        edge_arc_length = length / (nv - 2)
        nodes[:, 0] = (np.arange(nv) - 1.0) * edge_arc_length
        nominal_total_length = (nv - 1) * edge_arc_length
    else:
        edge_arc_length = length / (nv - 1)
        delta_theta = edge_arc_length / natural_radius
        theta = np.arange(nv) * delta_theta
        nodes[:, 0] = natural_radius * np.cos(theta)
        nodes[:, 1] = natural_radius * np.sin(theta)
        nominal_total_length = length

    return nodes, edge_arc_length, nominal_total_length


def _implicit_step(
    q_old,
    velocity_old,
    a1_old,
    reference_twist,
    free_indices,
    dt,
    tolerance,
    mass_vector,
    gravity_force,
    EA,
    EI,
    GJ,
    reference_lengths,
    voronoi_lengths,
    natural_curvature,
    natural_twist,
    maximum_iterations=100,
):
    """Advance one implicit-Euler step using Newton's method."""

    q = q_old.copy()
    for iteration in range(1, maximum_iterations + 1):
        a1_iterate, a2_iterate = computeTimeParallel(a1_old, q_old, q)
        tangent = computeTangent(q)
        reference_twist_iterate = getRefTwist(
            a1_iterate, tangent, reference_twist.copy()
        )
        theta = q[3::4]
        m1, m2 = computeMaterialDirectors(a1_iterate, a2_iterate, theta)

        bending_force, bending_jacobian = getFb(
            q, m1, m2, natural_curvature, EI, voronoi_lengths
        )
        twisting_force, twisting_jacobian = getFt(
            q,
            reference_twist_iterate,
            natural_twist,
            GJ,
            voronoi_lengths,
        )
        stretching_force, stretching_jacobian = getFs(
            q, EA, reference_lengths
        )

        elastic_force = bending_force + twisting_force + stretching_force
        elastic_jacobian = (
            bending_jacobian + twisting_jacobian + stretching_jacobian
        )
        velocity_new = (q - q_old) / dt
        residual = (
            mass_vector * ((velocity_new - velocity_old) / dt)
            - elastic_force
            - gravity_force
        )
        free_residual = residual[free_indices]
        if np.linalg.norm(free_residual) <= tolerance:
            return (
                q,
                velocity_new,
                a1_iterate,
                a2_iterate,
                reference_twist_iterate,
                iteration,
            )

        jacobian = np.diag(mass_vector / dt**2) - elastic_jacobian
        free_jacobian = jacobian[np.ix_(free_indices, free_indices)]
        q[free_indices] -= np.linalg.solve(free_jacobian, free_residual)

    raise RuntimeError(
        f"Newton's method failed after {maximum_iterations} iterations"
    )


def run_rod(
    nv=52,
    length=0.2,
    youngs_modulus=10.0e6,
    gravity=9.81,
    natural_radius=0.0,
    dt=0.1,
    total_time=10.0,
    remove_natural_curvature=False,
    radius=1.0e-3,
    poisson_ratio=0.5,
    density=1000.0,
):
    """Run one clamped-free DER case and return its final state and histories."""

    if nv < 3:
        raise ValueError("nv must be at least 3")
    if length <= 0.0 or radius <= 0.0 or youngs_modulus <= 0.0:
        raise ValueError("length, radius, and Young's modulus must be positive")
    if natural_radius < 0.0 or dt <= 0.0 or total_time <= 0.0:
        raise ValueError("natural_radius must be nonnegative; times must be positive")

    ne = nv - 1
    ndof = 4 * nv - 1
    nodes, edge_arc_length, nominal_total_length = _initial_geometry(
        nv, length, natural_radius
    )

    area = np.pi * radius**2
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    EA = youngs_modulus * area
    EI = youngs_modulus * np.pi * radius**4 / 4.0
    GJ = shear_modulus * np.pi * radius**4 / 2.0
    # The classroom value 1e-3*EI/L^2 is too loose for the small-gravity,
    # fine-mesh cases in Q2/Q4: their nodal loads approach the stopping force.
    # A tighter tolerance preserves the intended mesh-convergence trend.
    tolerance = 1.0e-6 * EI / length**2

    edge_mass = density * area * edge_arc_length
    mass_vector = np.zeros(ndof)
    for node in range(nv):
        position_indices = [4 * node, 4 * node + 1, 4 * node + 2]
        node_mass = edge_mass / 2.0 if node in (0, nv - 1) else edge_mass
        mass_vector[position_indices] = node_mass
    for edge in range(ne):
        mass_vector[4 * edge + 3] = 0.5 * edge_mass * radius**2
    gravity_force = np.zeros(ndof)
    gravity_vector = np.array([0.0, 0.0, -gravity])
    for node in range(nv):
        position_indices = [4 * node, 4 * node + 1, 4 * node + 2]
        gravity_force[position_indices] = (
            mass_vector[position_indices] * gravity_vector
        )

    q = np.zeros(ndof)
    for node in range(nv):
        q[4 * node : 4 * node + 3] = nodes[node]
    velocity = np.zeros(ndof)

    reference_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    voronoi_lengths = np.zeros(nv)
    voronoi_lengths[0] = 0.5 * reference_lengths[0]
    voronoi_lengths[-1] = 0.5 * reference_lengths[-1]
    voronoi_lengths[1:-1] = 0.5 * (
        reference_lengths[:-1] + reference_lengths[1:]
    )

    tangent = computeTangent(q)
    a1_first, _ = assignOrthonormalAdaptedFrame(tangent[0])
    a1, a2 = computeSpaceParallel(a1_first, q)
    m1, m2 = computeMaterialDirectors(a1, a2, q[3::4])
    reference_twist = np.zeros(nv)
    natural_curvature = getKappa(q, m1, m2)
    if remove_natural_curvature:
        natural_curvature = np.zeros_like(natural_curvature)
    natural_twist = np.zeros(nv)

    free_indices = np.arange(7, ndof)
    number_of_steps = round(total_time / dt)
    time = np.arange(number_of_steps + 1) * dt
    tip_z = np.empty(number_of_steps + 1)
    end_distance = np.empty(number_of_steps + 1)
    tip_z[0] = q[-1]
    end_distance[0] = np.linalg.norm(q[-3:] - q[:3])
    maximum_newton_iterations = 0

    for step in range(1, number_of_steps + 1):
        (
            q_new,
            velocity,
            a1,
            a2,
            reference_twist,
            iterations,
        ) = _implicit_step(
            q,
            velocity,
            a1,
            reference_twist,
            free_indices,
            dt,
            tolerance,
            mass_vector,
            gravity_force,
            EA,
            EI,
            GJ,
            reference_lengths,
            voronoi_lengths,
            natural_curvature,
            natural_twist,
        )
        q = q_new
        tip_z[step] = q[-1]
        end_distance[step] = np.linalg.norm(q[-3:] - q[:3])
        maximum_newton_iterations = max(maximum_newton_iterations, iterations)

    metric = end_distance if gravity == 0.0 else tip_z
    one_second_steps = min(number_of_steps, max(1, round(1.0 / dt)))
    change = abs(metric[-1] - metric[-1 - one_second_steps])
    steady_relative_change = change / max(abs(metric[-1]), 1.0e-14)

    # The nominal total length is intentionally computed above: for Part 1 it
    # includes the clamped first edge.  Keep this assertion near the mass setup.
    expected_mass = density * area * nominal_total_length
    if not np.isclose(np.sum(mass_vector[0::4]), expected_mass):
        raise RuntimeError("lumped translational masses do not match the rod mass")

    final_m1, final_m2 = computeMaterialDirectors(a1, a2, q[3::4])
    final_bending_force, _ = getFb(
        q, final_m1, final_m2, natural_curvature, EI, voronoi_lengths
    )
    final_twisting_force, _ = getFt(
        q,
        reference_twist,
        natural_twist,
        GJ,
        voronoi_lengths,
    )
    final_stretching_force, _ = getFs(q, EA, reference_lengths)
    static_residual = (
        final_bending_force
        + final_twisting_force
        + final_stretching_force
        + gravity_force
    )[free_indices]
    static_residual_norm = np.linalg.norm(static_residual)
    force_scale = max(
        np.linalg.norm(gravity_force[free_indices]),
        EI / length**2,
    )

    return RodResult(
        q=q,
        time=time,
        tip_z=tip_z,
        end_distance=end_distance,
        reference_lengths=reference_lengths,
        final_twist=q[3::4].copy(),
        steady_relative_change=steady_relative_change,
        maximum_newton_iterations=maximum_newton_iterations,
        final_velocity_norm=np.linalg.norm(velocity[free_indices]),
        free_static_residual_norm=static_residual_norm,
        relative_static_residual=static_residual_norm / force_scale,
    )


def _question_header(number, title):
    print("\n" + "=" * 72)
    print(f"Q{number}: {title}")


def main():
    """Run Q1-Q11 in order and print general data without choosing options."""

    cache = {}

    def cached_run(
        nv=52,
        youngs_modulus=10.0e6,
        gravity=9.81,
        natural_radius=0.0,
        remove_natural_curvature=False,
    ):
        key = (
            nv,
            float(youngs_modulus),
            float(gravity),
            float(natural_radius),
            bool(remove_natural_curvature),
        )
        if key not in cache:
            cache[key] = run_rod(
                nv=nv,
                youngs_modulus=youngs_modulus,
                gravity=gravity,
                natural_radius=natural_radius,
                remove_natural_curvature=remove_natural_curvature,
            )
        return cache[key]

    _question_header(1, "Euler-Bernoulli small-gravity prediction")
    eb_small = euler_bernoulli_deflection(10.0e6, 0.05)
    print(f"  delta_EB [m]                   {eb_small:.9g}")
    print(f"  delta_EB [mm]                  {1000.0 * eb_small:.9g}")

    _question_header(2, "DER small-load deflection")
    small_load = cached_run(gravity=0.05)
    print(f"  delta_DER [m]                  {-small_load.tip_z[-1]:.9g}")
    print(f"  delta_DER [mm]                 {-1000.0 * small_load.tip_z[-1]:.9g}")
    print(f"  last-second relative change    {small_load.steady_relative_change:.6%}")

    _question_header(3, "free length")
    dl = 0.2 / (52 - 2)
    print(f"  edge length [m]                {dl:.9g}")
    print(f"  free edges                     {52 - 2}")
    print(f"  free length [m]                {(52 - 2) * dl:.9g}")
    print(f"  all-edge length [m]            {(52 - 1) * dl:.9g}")
    print("  The first edge is fixed inside the clamp; free motion starts at node 1.")

    _question_header(4, "mesh convergence")
    print("  N      free edges      delta_DER [mm]      difference from EB")
    for nv in (22, 42, 82):
        result = cached_run(nv=nv, gravity=0.05)
        delta = -result.tip_z[-1]
        difference = (delta - eb_small) / eb_small
        print(f"  {nv:2d}     {nv - 2:5d}          {1000.0 * delta:12.9g}      {difference:12.6%}")

    _question_header(5, "stiffer rods under full gravity")
    print("  E [MPa]      delta_DER [m]      delta_EB [m]       e")
    for modulus_mpa in (10, 30, 100, 300, 1000):
        modulus = modulus_mpa * 1.0e6
        result = cached_run(youngs_modulus=modulus, gravity=9.81)
        delta_der = -result.tip_z[-1]
        delta_eb = euler_bernoulli_deflection(modulus, 9.81)
        relative_difference = abs(delta_der - delta_eb) / delta_eb
        print(
            f"  {modulus_mpa:7.0f}      {delta_der:13.9g}      "
            f"{delta_eb:12.9g}      {relative_difference:9.4%}"
        )

    _question_header(6, "heavier-load sweep")
    print("  g [m/s^2]    delta_DER [m]      delta_EB [m]       e")
    for gravity in (0.1, 0.3, 1.0, 3.0, 9.81):
        result = cached_run(youngs_modulus=10.0e6, gravity=gravity)
        delta_der = -result.tip_z[-1]
        delta_eb = euler_bernoulli_deflection(10.0e6, gravity)
        relative_difference = abs(delta_der - delta_eb) / delta_eb
        print(
            f"  {gravity:9.3g}    {delta_der:13.9g}      "
            f"{delta_eb:12.9g}      {relative_difference:9.4%}"
        )

    _question_header(7, "low-stiffness rod")
    default_straight = cached_run(youngs_modulus=10.0e6, gravity=9.81)
    print(f"  delta_DER [m]                  {-default_straight.tip_z[-1]:.9g}")
    print(f"  tip z [m]                      {default_straight.tip_z[-1]:.9g}")
    print(f"  last-second relative change    {default_straight.steady_relative_change:.6%}")

    _question_header(8, "large-deformation diagnostics")
    positions = node_positions(default_straight.q)
    final_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    edge_strain = final_lengths / default_straight.reference_lengths - 1.0
    delta_eb_default = euler_bernoulli_deflection(10.0e6, 9.81)
    print(f"  delta_DER [m]                  {-default_straight.tip_z[-1]:.9g}")
    print(f"  delta_EB [m]                   {delta_eb_default:.9g}")
    print(f"  final tip x [m]                {positions[-1, 0]:.9g}")
    print(f"  maximum |edge strain|          {np.max(np.abs(edge_strain)):.9g}")

    _question_header(9, "naturally curved rod")
    curved = cached_run(nv=50, natural_radius=0.02, gravity=9.81)
    print(f"  final tip z [m]                {curved.tip_z[-1]:.9g}")
    print(f"  last-second relative change    {curved.steady_relative_change:.6%}")

    _question_header(10, "straight-versus-curved twist")
    print(f"  straight max |theta| [rad]     {np.max(np.abs(default_straight.final_twist)):.9g}")
    print(f"  curved max |theta| [rad]       {np.max(np.abs(curved.final_twist)):.9g}")
    print(f"  straight tip z [m]             {default_straight.tip_z[-1]:.9g}")
    print(f"  curved tip z [m]               {curved.tip_z[-1]:.9g}")

    _question_header(11, "curved initial shape with zero natural curvature")
    zero_curvature = cached_run(
        nv=50,
        natural_radius=0.02,
        gravity=0.0,
        remove_natural_curvature=True,
    )
    print(f"  initial end distance [m]       {zero_curvature.end_distance[0]:.9g}")
    print(f"  final end distance [m]         {zero_curvature.end_distance[-1]:.9g}")
    print(f"  nominal arc length [m]         {0.2:.9g}")
    print(f"  last-second relative change    {zero_curvature.steady_relative_change:.6%}")


if __name__ == "__main__":
    main()
