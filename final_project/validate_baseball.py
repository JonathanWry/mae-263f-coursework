"""Quick checks for baseball geometry, pitcher impulse, and release speed."""

import sys

import numpy as np

sys.path.insert(0, "final_project")
from baseball_pitch import (  # noqa: E402
    pitcher_force,
    pitcher_spin_force,
    run_baseball,
)


def main() -> None:
    mass = 0.1453
    circumference = 0.5 * (9.0 + 9.25) * 0.0254
    radius = circumference / (2.0 * np.pi)
    assert 0.1417 <= mass <= 0.1488
    assert 0.0363 <= radius <= 0.0374

    duration = 0.080
    target_velocity = np.array([40.0, 0.0, 1.5])
    gravity = np.array([0.0, 0.0, -9.81])
    weights = np.array([0.2, 0.3, 0.5])
    sample_time = np.linspace(0.0, duration, 10001)
    total_force = np.array(
        [
            pitcher_force(t, duration, mass, target_velocity, gravity, weights)[0]
            .reshape(-1, 3)
            .sum(axis=0)
            for t in sample_time
        ]
    )
    impulse = np.trapezoid(total_force, sample_time, axis=0)
    expected_impulse = mass * (target_velocity - gravity * duration)
    np.testing.assert_allclose(impulse, expected_impulse, rtol=1.0e-8)

    spin_force, torque = pitcher_spin_force(
        time=0.01,
        duration=0.02,
        target_angular_velocity=np.array([0.0, -100.0, 0.0]),
        nodal_mass=np.full(4, 0.01),
        reference_offsets=np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        ),
    )
    np.testing.assert_allclose(spin_force.reshape(-1, 3).sum(axis=0), 0.0)
    assert torque > 0.0

    result = run_baseball(subdivisions=0, total_time=0.10, verbose=False)
    assert abs(result.release_speed - 40.0) < 0.5
    release_index = int(round(result.throw_duration / 0.001))
    assert (
        np.linalg.norm(result.center_velocity[-1])
        < np.linalg.norm(result.center_velocity[release_index])
    )

    print("Regulation mass and radius: passed")
    print("Pitcher-force impulse: passed")
    print("Pitcher torque has zero net force: passed")
    print(f"Release speed and aerodynamic slowdown: passed ({result.release_speed:.3f} m/s)")


if __name__ == "__main__":
    main()
