"""Predictor dictionary, cross-database comparison, and cohort-flow figure."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


OUTPUTS = rv.OUTPUTS
FIGURES = OUTPUTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

COLORS = {
    "mimic": "#DCE7F2",
    "eicu": "#DCE9E4",
    "sicdb": "#F3E1D8",
    "outline": "#4A4A4A",
    "text": "#222222",
    "muted": "#626262",
    "arrow": "#727272",
    "final": "#FFFFFF",
}


def predictor_dictionary() -> None:
    source_by_domain = {
        "demographic": {
            "mimic": "patients/icustays",
            "eicu": "patient",
            "sicdb": "cases.csv.gz",
        },
        "vital sign": {
            "mimic": "mimiciv_derived.vitalsign",
            "eicu": "vitalperiodic, vitalaperiodic, nursecharting",
            "sicdb": "data_float_h.csv.gz harmonized DataIDs",
        },
        "laboratory": {
            "mimic": "mimiciv_derived chemistry, CBC and blood-gas tables",
            "eicu": "lab",
            "sicdb": "laboratory.csv.gz harmonized DataIDs",
        },
        "landmark": {
            "mimic": "derived from ICU intime",
            "eicu": "derived from ICU offsets",
            "sicdb": "derived from ICUOffset",
        },
    }
    variable_labels = {
        "hr": ("Heart rate", "beats/min"),
        "sbp": ("Systolic blood pressure", "mmHg"),
        "dbp": ("Diastolic blood pressure", "mmHg"),
        "map": ("Mean arterial pressure", "mmHg"),
        "rr": ("Respiratory rate", "breaths/min"),
        "temp": ("Temperature", "degrees C"),
        "spo2": ("Peripheral oxygen saturation", "%"),
        "creatinine": ("Creatinine", "mg/dL"),
        "sodium": ("Sodium", "mmol/L"),
        "potassium": ("Potassium", "mmol/L"),
        "bicarbonate": ("Bicarbonate", "mmol/L"),
        "glucose": ("Glucose", "mg/dL"),
        "bun": ("Blood urea nitrogen", "mg/dL"),
        "lactate": ("Lactate", "mmol/L"),
        "ph": ("pH", "unitless"),
        "hemoglobin": ("Hemoglobin", "g/dL"),
        "hematocrit": ("Hematocrit", "%"),
        "wbc": ("White blood cell count", "10^9/L"),
    }
    rows = [
        {
            "feature": "age",
            "label": "Age at ICU admission",
            "domain": "demographic",
            "summary_rule": "Admission value",
            "lookback_window": "Not applicable",
            "unit": "years",
            "physiologic_range": "18 or older",
        },
        {
            "feature": "sex_male",
            "label": "Recorded sex",
            "domain": "demographic",
            "summary_rule": "Male=1, female=0; unavailable values missing",
            "lookback_window": "Not applicable",
            "unit": "binary",
            "physiologic_range": "0-1",
        },
        {
            "feature": "index_hour",
            "label": "Landmark hour",
            "domain": "landmark",
            "summary_rule": "Integer hour from ICU admission",
            "lookback_window": "ICU hours 6-24",
            "unit": "hours",
            "physiologic_range": "6-24",
        },
    ]
    suffix_rules = {
        "last": "Most recent non-missing value",
        "mean6": "Arithmetic mean",
        "min6": "Minimum",
        "max6": "Maximum",
    }
    for variable in rv.VITAL_VARS:
        label, unit = variable_labels[variable]
        low, high = rv.PHYSIOLOGIC_RANGES[variable]
        for suffix in ["last", "mean6", "min6", "max6"]:
            rows.append(
                {
                    "feature": f"{variable}_{suffix}",
                    "label": f"{label}: {suffix_rules[suffix].lower()}",
                    "domain": "vital sign",
                    "summary_rule": suffix_rules[suffix],
                    "lookback_window": "Previous 6 h through landmark",
                    "unit": unit,
                    "physiologic_range": f"{low:g}-{high:g}",
                }
            )
    for variable in rv.LAB_VARS:
        label, unit = variable_labels[variable]
        low, high = rv.PHYSIOLOGIC_RANGES[variable]
        rows.append(
            {
                "feature": f"{variable}_last",
                "label": f"{label}: most recent value",
                "domain": "laboratory",
                "summary_rule": "Most recent non-missing value",
                "lookback_window": "ICU admission through landmark",
                "unit": unit,
                "physiologic_range": f"{low:g}-{high:g}",
            }
        )

    frame = pd.DataFrame(rows)
    for database in ["mimic", "eicu", "sicdb"]:
        frame[f"source_{database}"] = frame["domain"].map(
            {domain: values[database] for domain, values in source_by_domain.items()}
        )
    frame["out_of_range_handling"] = "Set to missing before modeling"
    frame["hgb_missing_handling"] = "Native missing-value branch"
    frame["logistic_missing_handling"] = "Development median plus missingness indicator"
    if list(frame["feature"]) != rv.FEATURE_COLS:
        raise RuntimeError("Predictor dictionary order does not match model feature order")
    frame.to_csv(OUTPUTS / "corrected_predictor_dictionary.csv", index=False)

    lines = [
        "# Predictor dictionary",
        "",
        f"The frozen models used {len(frame)} predictors in the exact order listed below. Physiologically implausible values were set to missing before modeling using the same ranges in all databases.",
        "",
        "| Feature | Definition | Window | Unit | Accepted range |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.label} | {row.lookback_window} | {row.unit} | {row.physiologic_range} |"
        )
    lines.extend(
        [
            "",
            "HGB used native missing-value handling. Logistic regression used medians estimated in the MIMIC-IV development data and added missingness indicators. The detailed database source mappings are provided in `corrected_predictor_dictionary.csv`.",
        ]
    )
    (OUTPUTS / "corrected_predictor_dictionary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def cross_database_comparison() -> None:
    cohort = pd.read_csv(OUTPUTS / "corrected_table1_cohort_summary.csv").set_index("dataset")
    missing = pd.read_csv(OUTPUTS / "corrected_feature_missingness.csv").set_index("dataset")
    specs = [
        {
            "dataset_key": "mimic_temporal_test_2020_2022",
            "dataset": "MIMIC-IV temporal test",
            "release": "v3.1",
            "calendar_period": "2008-2022 source; anchor-year group 2020-2022 test",
            "setting": "Single US academic medical center",
            "centres_units": "1 center; multiple ICUs",
            "outcome_source": "mimiciv_derived.vasoactive_agent",
            "patient_linkage": "subject_id available",
        },
        {
            "dataset_key": "eicu_external",
            "dataset": "eICU-CRD external",
            "release": "v2.0",
            "calendar_period": "2014-2015",
            "setting": "US multicenter critical care",
            "centres_units": "208 hospitals; 335 units in source database",
            "outcome_source": "infusiondrug positive-rate records",
            "patient_linkage": "hospital-stay identifier; no permanent cross-admission person ID",
        },
        {
            "dataset_key": "sicdb_external",
            "dataset": "SICdb external",
            "release": "v1.0.8",
            "calendar_period": "2013-2021",
            "setting": "Single Austrian tertiary hospital",
            "centres_units": "1 center; CWIN and INBD primary units",
            "outcome_source": "medication.csv.gz continuous-use records",
            "patient_linkage": "PatientID available",
        },
    ]
    rows = []
    for spec in specs:
        key = spec.pop("dataset_key")
        c = cohort.loc[key]
        m = missing.loc[key]
        rows.append(
            {
                **spec,
                "n_stays": int(c.n_stays),
                "n_patients": int(c.n_patients),
                "age_median_iqr": f"{c.age_median:.0f} ({c.age_q1:.0f}-{c.age_q3:.0f})",
                "male_percent": float(c.male_percent),
                "event_positive_stay_percent": float(c.event_positive_stays_percent),
                "landmark_event_percent": float(c.positive_landmarks_percent),
                "rr_last_missing_percent": float(m.rr_last * 100),
                "temp_last_missing_percent": float(m.temp_last * 100),
                "lactate_last_missing_percent": float(m.lactate_last * 100),
                "ph_last_missing_percent": float(m.ph_last * 100),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUTS / "corrected_cross_database_comparison.csv", index=False)

    lines = [
        "# Cross-database design and measurement comparison",
        "",
        "| Characteristic | MIMIC-IV temporal test | eICU-CRD | SICdb |",
        "| --- | --- | --- | --- |",
    ]
    indexed = frame.set_index("dataset")
    columns = ["MIMIC-IV temporal test", "eICU-CRD external", "SICdb external"]
    display_rows = [
        ("Release", "release", "{}"),
        ("Calendar period", "calendar_period", "{}"),
        ("Setting", "setting", "{}"),
        ("Centers/units", "centres_units", "{}"),
        ("Eligible stays", "n_stays", "{:,}"),
        ("Unique patient identifiers", "n_patients", "{:,}"),
        ("Age, median (IQR), years", "age_median_iqr", "{}"),
        ("Recorded male sex, %", "male_percent", "{:.1f}"),
        ("Event-positive stays, %", "event_positive_stay_percent", "{:.2f}"),
        ("Positive landmarks, %", "landmark_event_percent", "{:.2f}"),
        ("Respiratory rate missing, %", "rr_last_missing_percent", "{:.1f}"),
        ("Temperature missing, %", "temp_last_missing_percent", "{:.1f}"),
        ("Lactate missing, %", "lactate_last_missing_percent", "{:.1f}"),
        ("pH missing, %", "ph_last_missing_percent", "{:.1f}"),
        ("Outcome record", "outcome_source", "{}"),
        ("Patient linkage", "patient_linkage", "{}"),
    ]
    for label, field, formatter in display_rows:
        values = [formatter.format(indexed.loc[column, field]) for column in columns]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "Missingness refers to the most recent value available under the feature definition and was measured before imputation. Differences reflect both patient mix and local measurement/documentation practice.",
        ]
    )
    (OUTPUTS / "corrected_cross_database_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def add_box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, color: str, final: bool = False) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=COLORS["final"] if final else color,
            edgecolor=color if final else COLORS["outline"],
            linewidth=1.2 if final else 0.8,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.2, color=COLORS["text"], linespacing=1.25)


def add_arrow(ax: plt.Axes, x: float, y_top: float, y_bottom: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x, y_top),
            (x, y_bottom),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=COLORS["arrow"],
        )
    )


def cohort_flow_figure() -> None:
    flow = pd.read_csv(OUTPUTS / "cohort_flow.csv").set_index("dataset")
    mimic = flow.loc["mimiciv"]
    eicu = flow.loc["eicu"]
    sicdb = flow.loc["sicdb_main_units"]

    columns = [
        {
            "x": 0.03,
            "title": "MIMIC-IV v3.1",
            "color": COLORS["mimic"],
            "steps": [
                f"Source adult ICU stays\n{int(mimic.source_adult_stays):,}",
                f"Length of stay >=12 h\n{int(mimic.adult_stays_los_ge_12h):,}",
                f"Eligible risk-set stays\n{int(mimic.risk_set_records):,}",
                f"Hourly landmarks\n{int(mimic.landmark_samples):,}\nPositive landmarks: {int(mimic.positive_landmarks):,}",
            ],
        },
        {
            "x": 0.355,
            "title": "eICU-CRD v2.0",
            "color": COLORS["eicu"],
            "steps": [
                f"Source adult unit stays\n{int(eicu.source_adult_unit_stays):,}",
                f"Length of stay >=12 h\n{int(eicu.adult_stays_los_ge_12h):,}",
                f"Eligible risk-set stays\n{int(eicu.risk_set_records):,}",
                f"Hourly landmarks\n{int(eicu.landmark_samples):,}\nPositive landmarks: {int(eicu.positive_landmarks):,}",
            ],
        },
        {
            "x": 0.68,
            "title": "SICdb v1.0.8",
            "color": COLORS["sicdb"],
            "steps": [
                f"Source adult cases\n{int(sicdb.source_adult_cases):,}",
                f"CWIN and INBD units\n{int(sicdb.adult_main_unit_cases):,}",
                f"Length of stay >=12 h\n{int(sicdb.adult_main_unit_los_ge_12h):,}",
                f"Eligible risk-set stays\n{int(sicdb.risk_set_records):,}",
                f"Hourly landmarks\n{int(sicdb.landmark_samples):,}\nPositive landmarks: {int(sicdb.positive_landmarks):,}",
            ],
        },
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.7))
    ax.set_axis_off()
    width = 0.285
    y_levels = [0.78, 0.61, 0.44, 0.27, 0.08]
    height = 0.11
    for column in columns:
        x = column["x"]
        ax.text(x + width / 2, 0.96, column["title"], ha="center", va="top", fontsize=9, fontweight="bold", color=COLORS["text"])
        if len(column["steps"]) == 4:
            levels = [0.76, 0.54, 0.32, 0.08]
        else:
            levels = y_levels
        for index, (level, text_value) in enumerate(zip(levels, column["steps"])):
            add_box(ax, x, level, width, height, text_value, column["color"], final=index == len(column["steps"]) - 1)
            if index < len(column["steps"]) - 1:
                next_level = levels[index + 1]
                add_arrow(ax, x + width / 2, level - 0.005, next_level + height + 0.005)
    ax.text(
        0.5,
        0.005,
        "Landmarks were sampled hourly from ICU hour 6 through hour 24; times after prior target-vasopressor initiation were excluded.",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["muted"],
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    name = FIGURES / "Figure_S2_cohort_flow"
    fig.savefig(name.with_suffix(".svg"), facecolor="white")
    fig.savefig(name.with_suffix(".pdf"), facecolor="white")
    fig.savefig(name.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(name.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    legend = (
        "**Extended Data Fig. 2 | Cohort construction across the three critical-care databases.** "
        "Counts show database-specific progression from source adult ICU or unit stays to the common hourly landmark risk set. "
        "Eligible landmarks were sampled from ICU hour 6 through hour 24, required observation through the subsequent six-hour horizon, "
        "and excluded times after a prior documented target-vasopressor start. SICdb primary validation was restricted to the CWIN and INBD units. "
        "Positive landmarks were followed by first continuous target-vasopressor initiation within six hours. Source data are provided in `cohort_flow.csv`."
    )
    (OUTPUTS / "corrected_flow_figure_legend.md").write_text(legend + "\n", encoding="utf-8")

    contract = """# Cohort-flow figure contract

- Core conclusion: database-specific eligibility paths converge on the same hourly landmark risk-set definition.
- Evidence chain: source adult stays/cases -> duration and unit restrictions -> eligible risk-set stays -> hourly landmarks and positives.
- Archetype: schematic-led composite with three parallel database columns.
- Backend: Python/matplotlib only.
- Export contract: 180-mm-wide white-background figure; editable SVG/PDF plus 300-dpi PNG and 600-dpi TIFF; aggregate counts only.
- Reviewer risks: counts must match `cohort_flow.csv`; SICdb must not be labelled multicenter; landmarks must not be described as independent patients.
"""
    (rv.WORK / "flow_figure_contract.md").write_text(contract, encoding="utf-8")


def main() -> None:
    predictor_dictionary()
    cross_database_comparison()
    cohort_flow_figure()


if __name__ == "__main__":
    main()
