"""Generate coupled-model result and parameter-sweep figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from coupled_archery import (  # noqa: E402
    plot_coupled_results,
    run_coupled_archery,
    run_parameter_sweep,
)


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "midterm_report" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    result = run_coupled_archery(verbose=False)
    result_figure = plot_coupled_results(result)
    result_figure.savefig(output_directory / "coupled_results.pdf", bbox_inches="tight")
    result_figure.savefig(
        output_directory / "coupled_results.png", dpi=180, bbox_inches="tight"
    )
    plt.close(result_figure)

    rows = run_parameter_sweep()
    parameters = [
        "string_EA",
        "string_mass",
        "string_damping",
        "limb_flexural_rigidity",
    ]
    labels = {
        "string_EA": "string axial stiffness EA",
        "string_mass": "string mass",
        "string_damping": "string damping",
        "limb_flexural_rigidity": "limb flexural rigidity EI",
    }
    sweep_figure, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for axis, parameter in zip(axes.ravel(), parameters):
        selected = [row for row in rows if row["parameter"] == parameter]
        factors = [float(row["factor"]) for row in selected]
        speeds = [float(row["release_speed_m_s"]) for row in selected]
        efficiencies = [float(row["efficiency_percent"]) for row in selected]
        speed_line = axis.plot(
            factors, speeds, "o-", color="tab:blue", label="release speed"
        )
        axis.set_xlabel("parameter / baseline value [-]")
        axis.set_ylabel("release speed [m/s]", color="tab:blue")
        axis.tick_params(axis="y", labelcolor="tab:blue")
        efficiency_axis = axis.twinx()
        efficiency_line = efficiency_axis.plot(
            factors,
            efficiencies,
            "s--",
            color="tab:orange",
            label="efficiency",
        )
        efficiency_axis.set_ylabel("launch efficiency [%]", color="tab:orange")
        efficiency_axis.tick_params(axis="y", labelcolor="tab:orange")
        axis.set_title(labels[parameter])
        axis.grid(alpha=0.3)
        lines = speed_line + efficiency_line
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8)
    sweep_figure.tight_layout()
    sweep_figure.savefig(output_directory / "coupled_sweep.pdf", bbox_inches="tight")
    sweep_figure.savefig(
        output_directory / "coupled_sweep.png", dpi=180, bbox_inches="tight"
    )
    plt.close(sweep_figure)

    print(output_directory / "coupled_results.pdf")
    print(output_directory / "coupled_sweep.pdf")
    for row in rows:
        print(
            f"{row['parameter']:26s} factor={float(row['factor']):.2f} "
            f"v={float(row['release_speed_m_s']):.3f} m/s "
            f"eta={float(row['efficiency_percent']):.2f}%"
        )


if __name__ == "__main__":
    main()
