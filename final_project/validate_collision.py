"""Quick geometry, equilibrium, trajectory, and collision checks."""

import numpy as np

from soft_shell_collision import (
    COURSE_SHELL_DIR,
    make_icosphere,
    run_simulation,
)

import sys

sys.path.insert(0, str(COURSE_SHELL_DIR))
from shell_functions import getEdgesAndHinges, getFb, getFs, getThetaBar  # noqa: E402


def main() -> None:
    radius = 0.10
    center = np.array([0.0, 0.0, 1.0])
    nodes, elements = make_icosphere(radius, center, subdivisions=0)
    edges, hinges = getEdgesAndHinges(elements)

    # A closed triangular sphere must obey Euler's polyhedron relationships.
    assert len(elements) == 2 * len(nodes) - 4
    assert len(edges) == 3 * len(nodes) - 6
    assert len(hinges) == len(edges)

    q = nodes.reshape(-1)
    reference_lengths = np.linalg.norm(
        nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1
    )
    youngs_modulus = 2.0e5
    thickness = 0.005
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
    stretching_force, _ = getFs(
        q, edges, reference_lengths, stretching_stiffness
    )
    bending_force, _ = getFb(q, hinges, natural_angles, bending_stiffness)
    assert np.linalg.norm(stretching_force, ord=np.inf) < 1.0e-10
    assert np.linalg.norm(bending_force, ord=np.inf) < 1.0e-10

    result = run_simulation(
        subdivisions=0,
        dt=0.004,
        total_time=0.60,
        output_stride=5,
        verbose=False,
    )

    # Before contact, backward Euler under constant gravity has this exact
    # discrete trajectory: z_n=z_0+v_0*t+0.5*a*(t^2+t*dt).
    before_contact = result.time <= 0.25
    t = result.time[before_contact]
    expected_x = 2.5 * t
    expected_z = 1.0 + 0.2 * t - 0.5 * 9.81 * (t**2 + t * 0.004)
    np.testing.assert_allclose(
        result.center_of_mass[before_contact, 0], expected_x, atol=2.0e-10
    )
    np.testing.assert_allclose(
        result.center_of_mass[before_contact, 2], expected_z, atol=2.0e-10
    )

    assert result.wall_hit_time is not None
    assert result.ground_hit_time is not None
    assert result.wall_hit_time < result.ground_hit_time
    assert result.wall_penetration.max() < 0.01
    assert result.ground_penetration.max() < 0.01

    print("Geometry and zero-force reference state: passed")
    print("Free-flight center-of-mass trajectory: passed")
    print(
        "Collision order and penetration: passed "
        f"(wall {result.wall_hit_time:.3f} s, "
        f"ground {result.ground_hit_time:.3f} s)"
    )


if __name__ == "__main__":
    main()
