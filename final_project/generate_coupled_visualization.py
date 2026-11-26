"""Generate the inline coupled-launch visualization from simulation states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coupled_archery import (
    CoupledBowParameters,
    nodes_and_mapping,
    run_coupled_archery,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def rounded(value: float) -> float:
    return round(float(value), 6)


def main() -> None:
    args = parse_arguments()
    parameters = CoupledBowParameters()
    result = run_coupled_archery(
        parameters,
        force_draw_points=13,
        compute_flight=False,
        verbose=False,
    )
    frame_indices = np.unique(
        np.rint(np.linspace(0, len(result.time) - 1, 101)).astype(int)
    )
    release_index = int(np.searchsorted(result.time, result.release_time))
    frame_indices = np.unique(np.r_[frame_indices, release_index - 1, release_index])
    frames = []
    for index in frame_indices:
        state = result.states[int(index)]
        nodes, _, _, _ = nodes_and_mapping(state, parameters)
        simulation_time = float(result.time[index])
        arrow_x = float(state[parameters.arrow_x_index])
        frames.append(
            {
                "t": rounded(simulation_time),
                "nodes": [[rounded(x), rounded(y)] for x, y in nodes],
                "arrowX": rounded(arrow_x),
                "speed": rounded(result.arrow_speed[index]),
                "force": rounded(result.contact_force[index]),
            }
        )

    payload = {
        "braceX": rounded(result.equilibria.brace_x),
        "releaseTime": rounded(result.release_time),
        "releasePosition": rounded(result.release_position),
        "arrowLength": rounded(parameters.arrow_length),
        "frames": frames,
    }
    template_path = Path(__file__).with_name("coupled-launch-template.html")
    fragment = template_path.read_text(encoding="utf-8")
    fragment = fragment.replace(
        "__SIMULATION_DATA__",
        json.dumps(payload, separators=(",", ":")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
