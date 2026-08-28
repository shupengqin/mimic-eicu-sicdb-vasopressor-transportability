# Outcome harmonization audit

The target was the first documented start of a continuous norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine infusion during the ICU stay. Landmark rows after a prior target start were excluded.

| Database | Source | Continuous-use rule | Start time | Drug mapping |
| --- | --- | --- | --- | --- |
| MIMIC-IV | mimiciv_derived.vasoactive_agent | At least one target agent rate greater than zero | Earliest starttime | Named columns for the five target agents |
| eICU-CRD | infusiondrug | Numeric drugrate or infusionrate greater than zero | Earliest infusionoffset | Drug-name regex including generic and common synonym names |
| SICdb | medication.csv.gz | IsSingleDose = 0 | Earliest medication Offset corrected by ICUOffset | DrugIDs 1502 epinephrine, 1550 vasopressin, 1562 norepinephrine, 1593 phenylephrine, 1618 dopamine |

Database records establish documented treatment initiation, not physiologic shock onset or clinician intent. Differences in medication charting remain a potential source of outcome misclassification.
