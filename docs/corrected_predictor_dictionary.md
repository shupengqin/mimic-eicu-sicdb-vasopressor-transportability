# Predictor dictionary

The frozen models used 42 predictors in the exact order listed below. Physiologically implausible values were set to missing before modeling using the same ranges in all databases.

| Feature | Definition | Window | Unit | Accepted range |
| --- | --- | --- | --- | --- |
| `age` | Age at ICU admission | Not applicable | years | 18 or older |
| `sex_male` | Recorded sex | Not applicable | binary | 0-1 |
| `index_hour` | Landmark hour | ICU hours 6-24 | hours | 6-24 |
| `hr_last` | Heart rate: most recent non-missing value | Previous 6 h through landmark | beats/min | 20-250 |
| `hr_mean6` | Heart rate: arithmetic mean | Previous 6 h through landmark | beats/min | 20-250 |
| `hr_min6` | Heart rate: minimum | Previous 6 h through landmark | beats/min | 20-250 |
| `hr_max6` | Heart rate: maximum | Previous 6 h through landmark | beats/min | 20-250 |
| `sbp_last` | Systolic blood pressure: most recent non-missing value | Previous 6 h through landmark | mmHg | 30-300 |
| `sbp_mean6` | Systolic blood pressure: arithmetic mean | Previous 6 h through landmark | mmHg | 30-300 |
| `sbp_min6` | Systolic blood pressure: minimum | Previous 6 h through landmark | mmHg | 30-300 |
| `sbp_max6` | Systolic blood pressure: maximum | Previous 6 h through landmark | mmHg | 30-300 |
| `dbp_last` | Diastolic blood pressure: most recent non-missing value | Previous 6 h through landmark | mmHg | 10-200 |
| `dbp_mean6` | Diastolic blood pressure: arithmetic mean | Previous 6 h through landmark | mmHg | 10-200 |
| `dbp_min6` | Diastolic blood pressure: minimum | Previous 6 h through landmark | mmHg | 10-200 |
| `dbp_max6` | Diastolic blood pressure: maximum | Previous 6 h through landmark | mmHg | 10-200 |
| `map_last` | Mean arterial pressure: most recent non-missing value | Previous 6 h through landmark | mmHg | 20-250 |
| `map_mean6` | Mean arterial pressure: arithmetic mean | Previous 6 h through landmark | mmHg | 20-250 |
| `map_min6` | Mean arterial pressure: minimum | Previous 6 h through landmark | mmHg | 20-250 |
| `map_max6` | Mean arterial pressure: maximum | Previous 6 h through landmark | mmHg | 20-250 |
| `rr_last` | Respiratory rate: most recent non-missing value | Previous 6 h through landmark | breaths/min | 2-100 |
| `rr_mean6` | Respiratory rate: arithmetic mean | Previous 6 h through landmark | breaths/min | 2-100 |
| `rr_min6` | Respiratory rate: minimum | Previous 6 h through landmark | breaths/min | 2-100 |
| `rr_max6` | Respiratory rate: maximum | Previous 6 h through landmark | breaths/min | 2-100 |
| `temp_last` | Temperature: most recent non-missing value | Previous 6 h through landmark | degrees C | 25-45 |
| `temp_mean6` | Temperature: arithmetic mean | Previous 6 h through landmark | degrees C | 25-45 |
| `temp_min6` | Temperature: minimum | Previous 6 h through landmark | degrees C | 25-45 |
| `temp_max6` | Temperature: maximum | Previous 6 h through landmark | degrees C | 25-45 |
| `spo2_last` | Peripheral oxygen saturation: most recent non-missing value | Previous 6 h through landmark | % | 50-100 |
| `spo2_mean6` | Peripheral oxygen saturation: arithmetic mean | Previous 6 h through landmark | % | 50-100 |
| `spo2_min6` | Peripheral oxygen saturation: minimum | Previous 6 h through landmark | % | 50-100 |
| `spo2_max6` | Peripheral oxygen saturation: maximum | Previous 6 h through landmark | % | 50-100 |
| `creatinine_last` | Creatinine: most recent value | ICU admission through landmark | mg/dL | 0.05-30 |
| `sodium_last` | Sodium: most recent value | ICU admission through landmark | mmol/L | 80-200 |
| `potassium_last` | Potassium: most recent value | ICU admission through landmark | mmol/L | 1-12 |
| `bicarbonate_last` | Bicarbonate: most recent value | ICU admission through landmark | mmol/L | 5-60 |
| `glucose_last` | Glucose: most recent value | ICU admission through landmark | mg/dL | 20-1000 |
| `bun_last` | Blood urea nitrogen: most recent value | ICU admission through landmark | mg/dL | 1-200 |
| `lactate_last` | Lactate: most recent value | ICU admission through landmark | mmol/L | 0.1-30 |
| `ph_last` | pH: most recent value | ICU admission through landmark | unitless | 6.5-8 |
| `hemoglobin_last` | Hemoglobin: most recent value | ICU admission through landmark | g/dL | 3-25 |
| `hematocrit_last` | Hematocrit: most recent value | ICU admission through landmark | % | 10-80 |
| `wbc_last` | White blood cell count: most recent value | ICU admission through landmark | 10^9/L | 0.1-300 |

HGB used native missing-value handling. Logistic regression used medians estimated in the MIMIC-IV development data and added missingness indicators. The detailed database source mappings are provided in `corrected_predictor_dictionary.csv`.
