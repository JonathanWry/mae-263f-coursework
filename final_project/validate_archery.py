"""Quick physical and numerical checks for the archery predictor."""

import sys

import numpy as np

sys.path.insert(0, "final_project")
from archery_prediction import (  # noqa: E402
    assemble_string_elasticity,
    effective_axial_stiffness,
    initial_string_nodes,
    run_archery,
)


def main() -> None:
    half_span = 0.60
    draw_length = 0.70
    requested_force = 200.0
    nodes = initial_string_nodes(21, half_span, draw_length)
    rest_length = 2.0 * half_span / 20
    EA = effective_axial_stiffness(requested_force, half_span, draw_length)
    _, gradient, _ = assemble_string_elasticity(
        nodes.reshape(-1), rest_length, EA
    )
    midpoint_force = -gradient[2 * 10]
    np.testing.assert_allclose(midpoint_force, requested_force, rtol=1.0e-12)

    coarse = run_archery(node_count=11, verbose=False)
    default = run_archery(node_count=21, verbose=False)
    fine = run_archery(node_count=41, verbose=False)
    assert default.release_speed < default.energy_limit_speed
    assert default.release_speed > 0.90 * default.energy_limit_speed
    assert max(
        abs(coarse.release_speed - default.release_speed),
        abs(fine.release_speed - default.release_speed),
    ) < 0.30
    assert abs(default.target_height_error) < 1.0e-5
    assert 0.0 < default.required_aim_angle < 20.0

    print("Calibrated full-draw force: passed")
    print("Arrow energy bound: passed")
    print(
        "Mesh convergence: passed "
        f"({coarse.release_speed:.3f}, {default.release_speed:.3f}, "
        f"{fine.release_speed:.3f} m/s)"
    )
    print("70 m trajectory prediction: passed")


if __name__ == "__main__":
    main()
