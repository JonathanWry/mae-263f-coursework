"""Generate reproducible figures for the MAE 263F midterm report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from archery_prediction import plot_archery_results, run_archery  # noqa: E402


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "midterm_report" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    default = run_archery(verbose=False)
    results_figure = plot_archery_results(default)
    results_figure.savefig(
        output_directory / "launch_results.pdf",
        bbox_inches="tight",
    )
    plt.close(results_figure)

    node_counts = np.array([11, 21, 41])
    mesh_speeds = np.array(
        [run_archery(node_count=int(count), verbose=False).release_speed for count in node_counts]
    )
    time_steps = np.array([2.0e-4, 1.0e-4, 5.0e-5])
    time_step_speeds = np.array(
        [run_archery(launch_dt=float(dt), verbose=False).release_speed for dt in time_steps]
    )

    convergence_figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(node_counts, mesh_speeds, "o-", label="release speed")
    axes[0].set_xlabel("number of string nodes, N [-]")
    axes[0].set_ylabel("release speed [m/s]")
    axes[0].set_xticks(node_counts)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(1.0e3 * time_steps, time_step_speeds, "o-", label="release speed")
    axes[1].set_xlabel("launch time step [ms]")
    axes[1].set_ylabel("release speed [m/s]")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    convergence_figure.tight_layout()
    convergence_figure.savefig(
        output_directory / "convergence.pdf",
        bbox_inches="tight",
    )
    plt.close(convergence_figure)

    print(output_directory / "launch_results.pdf")
    print(output_directory / "convergence.pdf")


if __name__ == "__main__":
    main()
