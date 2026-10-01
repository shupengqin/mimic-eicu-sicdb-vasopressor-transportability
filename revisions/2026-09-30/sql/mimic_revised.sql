-- Prediction-time-identifiable hourly risk set for MIMIC-IV v3.1.
-- A landmark is retained when the stay is still observed at that hour and no
-- target pressor has been documented before it.  Future six-hour observability
-- is not used to decide eligibility.  Run with psql and redirect STDOUT to a
-- local CSV; the CSV must remain under the data custodian's access controls.
\set ON_ERROR_STOP on
SET statement_timeout = 0;
SET lock_timeout = 0;
SET work_mem = '256MB';
SET temp_buffers = '128MB';

DROP TABLE IF EXISTS pg_temp.mimic_base;
CREATE TEMP TABLE mimic_base AS
SELECT x.*
FROM (
    SELECT
        i.stay_id,
        i.subject_id,
        i.hadm_id,
        i.first_careunit,
        i.intime,
        i.outtime,
        EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 AS los_hours,
        EXTRACT(YEAR FROM i.intime)::int AS time_year,
        p.anchor_year_group::text AS time_group,
        p.anchor_year,
        p.dod,
        p.gender,
        (p.anchor_age + EXTRACT(YEAR FROM i.intime)::int - p.anchor_year)::double precision AS age
    FROM mimiciv_icu.icustays AS i
    JOIN mimiciv_hosp.patients AS p USING (subject_id)
    WHERE (p.anchor_age + EXTRACT(YEAR FROM i.intime)::int - p.anchor_year) >= 18
      AND i.outtime > i.intime + interval '6 hours'
) AS x;
CREATE INDEX mimic_base_stay_idx ON mimic_base(stay_id);
ANALYZE mimic_base;

DROP TABLE IF EXISTS pg_temp.mimic_pressor;
CREATE TEMP TABLE mimic_pressor AS
SELECT
    v.stay_id,
    MIN(v.starttime) AS first_start
FROM mimiciv_derived.vasoactive_agent AS v
WHERE COALESCE(v.norepinephrine, 0) > 0
   OR COALESCE(v.epinephrine, 0) > 0
   OR COALESCE(v.phenylephrine, 0) > 0
   OR COALESCE(v.vasopressin, 0) > 0
   OR COALESCE(v.dopamine, 0) > 0
GROUP BY v.stay_id;
CREATE INDEX mimic_pressor_stay_idx ON mimic_pressor(stay_id);
ANALYZE mimic_pressor;

DROP TABLE IF EXISTS pg_temp.mimic_grid;
CREATE TEMP TABLE mimic_grid AS
SELECT
    b.stay_id,
    b.subject_id,
    b.hadm_id,
    b.first_careunit AS unit_name,
    b.intime,
    b.outtime,
    b.los_hours,
    b.time_year,
    b.time_group,
    b.anchor_year,
    b.dod,
    b.gender,
    b.age,
    g.index_hour,
    b.intime + (g.index_hour * interval '1 hour') AS index_time,
    p.first_start
FROM mimic_base AS b
LEFT JOIN mimic_pressor AS p USING (stay_id)
CROSS JOIN LATERAL generate_series(6, 24) AS g(index_hour)
WHERE b.intime + (g.index_hour * interval '1 hour') < b.outtime
  AND (p.first_start IS NULL OR p.first_start > b.intime + (g.index_hour * interval '1 hour'));
CREATE INDEX mimic_grid_stay_idx ON mimic_grid(stay_id);
CREATE INDEX mimic_grid_time_idx ON mimic_grid(stay_id,index_time);
ANALYZE mimic_grid;

DROP TABLE IF EXISTS pg_temp.mimic_vitals;
CREATE TEMP TABLE mimic_vitals AS
SELECT
    v.stay_id,
    EXTRACT(EPOCH FROM (v.charttime - b.intime)) / 3600.0 AS rel_hour,
    v.heart_rate::double precision AS hr,
    v.sbp::double precision AS sbp,
    v.dbp::double precision AS dbp,
    v.mbp::double precision AS map,
    v.resp_rate::double precision AS rr,
    v.temperature::double precision AS temp,
    v.spo2::double precision AS spo2
FROM mimiciv_derived.vitalsign AS v
JOIN mimic_base AS b USING (stay_id)
WHERE v.charttime > b.intime
  AND v.charttime <= b.intime + interval '24 hours';
\copy (SELECT 'mimic' AS dataset, 'hr' AS variable, count(hr) AS nonmissing_observations, count(*) FILTER (WHERE hr IS NOT NULL AND NOT (hr BETWEEN 20 AND 250)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'sbp' AS variable, count(sbp) AS nonmissing_observations, count(*) FILTER (WHERE sbp IS NOT NULL AND NOT (sbp BETWEEN 30 AND 300)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'dbp' AS variable, count(dbp) AS nonmissing_observations, count(*) FILTER (WHERE dbp IS NOT NULL AND NOT (dbp BETWEEN 10 AND 200)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'map' AS variable, count(map) AS nonmissing_observations, count(*) FILTER (WHERE map IS NOT NULL AND NOT (map BETWEEN 20 AND 250)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'rr' AS variable, count(rr) AS nonmissing_observations, count(*) FILTER (WHERE rr IS NOT NULL AND NOT (rr BETWEEN 2 AND 100)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'temp' AS variable, count(temp) AS nonmissing_observations, count(*) FILTER (WHERE temp IS NOT NULL AND NOT (temp BETWEEN 25 AND 45)) AS rejected_observations FROM mimic_vitals UNION ALL SELECT 'mimic' AS dataset, 'spo2' AS variable, count(spo2) AS nonmissing_observations, count(*) FILTER (WHERE spo2 IS NOT NULL AND NOT (spo2 BETWEEN 50 AND 100)) AS rejected_observations FROM mimic_vitals) TO 'results/mimic_source_qc_mimic_vitals.csv' WITH CSV HEADER
UPDATE mimic_vitals SET hr=CASE WHEN hr BETWEEN 20 AND 250 THEN hr END, sbp=CASE WHEN sbp BETWEEN 30 AND 300 THEN sbp END, dbp=CASE WHEN dbp BETWEEN 10 AND 200 THEN dbp END, map=CASE WHEN map BETWEEN 20 AND 250 THEN map END, rr=CASE WHEN rr BETWEEN 2 AND 100 THEN rr END, temp=CASE WHEN temp BETWEEN 25 AND 45 THEN temp END, spo2=CASE WHEN spo2 BETWEEN 50 AND 100 THEN spo2 END;
CREATE INDEX mimic_vitals_stay_hour_idx ON mimic_vitals(stay_id,rel_hour);
ANALYZE mimic_vitals;

DROP TABLE IF EXISTS pg_temp.mimic_chem;
CREATE TEMP TABLE mimic_chem AS
SELECT
    b.stay_id,
    EXTRACT(EPOCH FROM (c.charttime - b.intime)) / 3600.0 AS rel_hour,
    c.creatinine::double precision AS creatinine,
    c.sodium::double precision AS sodium,
    c.potassium::double precision AS potassium,
    c.bicarbonate::double precision AS bicarbonate,
    c.glucose::double precision AS glucose,
    c.bun::double precision AS bun
FROM mimiciv_derived.chemistry AS c
JOIN mimic_base AS b ON b.subject_id = c.subject_id AND b.hadm_id = c.hadm_id
WHERE c.charttime > b.intime
  AND c.charttime <= b.intime + interval '24 hours';
\copy (SELECT 'mimic' AS dataset, 'creatinine' AS variable, count(creatinine) AS nonmissing_observations, count(*) FILTER (WHERE creatinine IS NOT NULL AND NOT (creatinine BETWEEN 0.05 AND 30)) AS rejected_observations FROM mimic_chem UNION ALL SELECT 'mimic' AS dataset, 'sodium' AS variable, count(sodium) AS nonmissing_observations, count(*) FILTER (WHERE sodium IS NOT NULL AND NOT (sodium BETWEEN 80 AND 200)) AS rejected_observations FROM mimic_chem UNION ALL SELECT 'mimic' AS dataset, 'potassium' AS variable, count(potassium) AS nonmissing_observations, count(*) FILTER (WHERE potassium IS NOT NULL AND NOT (potassium BETWEEN 1 AND 12)) AS rejected_observations FROM mimic_chem UNION ALL SELECT 'mimic' AS dataset, 'bicarbonate' AS variable, count(bicarbonate) AS nonmissing_observations, count(*) FILTER (WHERE bicarbonate IS NOT NULL AND NOT (bicarbonate BETWEEN 5 AND 60)) AS rejected_observations FROM mimic_chem UNION ALL SELECT 'mimic' AS dataset, 'glucose' AS variable, count(glucose) AS nonmissing_observations, count(*) FILTER (WHERE glucose IS NOT NULL AND NOT (glucose BETWEEN 20 AND 1000)) AS rejected_observations FROM mimic_chem UNION ALL SELECT 'mimic' AS dataset, 'bun' AS variable, count(bun) AS nonmissing_observations, count(*) FILTER (WHERE bun IS NOT NULL AND NOT (bun BETWEEN 1 AND 200)) AS rejected_observations FROM mimic_chem) TO 'results/mimic_source_qc_mimic_chem.csv' WITH CSV HEADER
UPDATE mimic_chem SET creatinine=CASE WHEN creatinine BETWEEN 0.05 AND 30 THEN creatinine END, sodium=CASE WHEN sodium BETWEEN 80 AND 200 THEN sodium END, potassium=CASE WHEN potassium BETWEEN 1 AND 12 THEN potassium END, bicarbonate=CASE WHEN bicarbonate BETWEEN 5 AND 60 THEN bicarbonate END, glucose=CASE WHEN glucose BETWEEN 20 AND 1000 THEN glucose END, bun=CASE WHEN bun BETWEEN 1 AND 200 THEN bun END;
CREATE INDEX mimic_chem_stay_hour_idx ON mimic_chem(stay_id,rel_hour);
ANALYZE mimic_chem;

DROP TABLE IF EXISTS pg_temp.mimic_cbc;
CREATE TEMP TABLE mimic_cbc AS
SELECT
    b.stay_id,
    EXTRACT(EPOCH FROM (c.charttime - b.intime)) / 3600.0 AS rel_hour,
    c.hemoglobin::double precision AS hemoglobin,
    c.hematocrit::double precision AS hematocrit,
    c.wbc::double precision AS wbc
FROM mimiciv_derived.complete_blood_count AS c
JOIN mimic_base AS b ON b.subject_id = c.subject_id AND b.hadm_id = c.hadm_id
WHERE c.charttime > b.intime
  AND c.charttime <= b.intime + interval '24 hours';
\copy (SELECT 'mimic' AS dataset, 'hemoglobin' AS variable, count(hemoglobin) AS nonmissing_observations, count(*) FILTER (WHERE hemoglobin IS NOT NULL AND NOT (hemoglobin BETWEEN 3 AND 25)) AS rejected_observations FROM mimic_cbc UNION ALL SELECT 'mimic' AS dataset, 'hematocrit' AS variable, count(hematocrit) AS nonmissing_observations, count(*) FILTER (WHERE hematocrit IS NOT NULL AND NOT (hematocrit BETWEEN 10 AND 80)) AS rejected_observations FROM mimic_cbc UNION ALL SELECT 'mimic' AS dataset, 'wbc' AS variable, count(wbc) AS nonmissing_observations, count(*) FILTER (WHERE wbc IS NOT NULL AND NOT (wbc BETWEEN 0.1 AND 300)) AS rejected_observations FROM mimic_cbc) TO 'results/mimic_source_qc_mimic_cbc.csv' WITH CSV HEADER
UPDATE mimic_cbc SET hemoglobin=CASE WHEN hemoglobin BETWEEN 3 AND 25 THEN hemoglobin END, hematocrit=CASE WHEN hematocrit BETWEEN 10 AND 80 THEN hematocrit END, wbc=CASE WHEN wbc BETWEEN 0.1 AND 300 THEN wbc END;
CREATE INDEX mimic_cbc_stay_hour_idx ON mimic_cbc(stay_id,rel_hour);
ANALYZE mimic_cbc;

DROP TABLE IF EXISTS pg_temp.mimic_bg;
CREATE TEMP TABLE mimic_bg AS
SELECT
    b.stay_id,
    EXTRACT(EPOCH FROM (g.charttime - b.intime)) / 3600.0 AS rel_hour,
    g.lactate::double precision AS lactate,
    g.ph::double precision AS ph
FROM mimiciv_derived.bg AS g
JOIN mimic_base AS b ON b.subject_id = g.subject_id AND b.hadm_id = g.hadm_id
WHERE g.charttime > b.intime
  AND g.charttime <= b.intime + interval '24 hours';
\copy (SELECT 'mimic' AS dataset, 'lactate' AS variable, count(lactate) AS nonmissing_observations, count(*) FILTER (WHERE lactate IS NOT NULL AND NOT (lactate BETWEEN 0.1 AND 30)) AS rejected_observations FROM mimic_bg UNION ALL SELECT 'mimic' AS dataset, 'ph' AS variable, count(ph) AS nonmissing_observations, count(*) FILTER (WHERE ph IS NOT NULL AND NOT (ph BETWEEN 6.5 AND 8)) AS rejected_observations FROM mimic_bg) TO 'results/mimic_source_qc_mimic_bg.csv' WITH CSV HEADER
UPDATE mimic_bg SET lactate=CASE WHEN lactate BETWEEN 0.1 AND 30 THEN lactate END, ph=CASE WHEN ph BETWEEN 6.5 AND 8 THEN ph END;
CREATE INDEX mimic_bg_stay_hour_idx ON mimic_bg(stay_id,rel_hour);
ANALYZE mimic_bg;

COPY (
SELECT
    'mimiciv'::text AS dataset,
    s.stay_id::bigint AS record_id,
    s.subject_id::bigint AS patient_id,
    s.unit_name::text AS unit_name,
    s.time_year::int AS time_year,
    s.time_group::text AS time_group,
    s.time_year-s.anchor_year AS anchor_year_delta,
    (s.dod IS NOT NULL AND s.dod=s.outtime::date)::int AS death_on_exit_date,
    s.index_hour::int AS index_hour,
    s.age::double precision AS age,
    CASE WHEN upper(s.gender) = 'M' THEN 1.0 WHEN upper(s.gender) = 'F' THEN 0.0 ELSE NULL END AS sex_male,
    CASE WHEN s.first_start IS NOT NULL
              AND s.first_start < LEAST(s.index_time + interval '6 hours', s.outtime)
         THEN 1 ELSE 0 END AS label,
    CASE WHEN s.first_start IS NOT NULL
              AND s.first_start < LEAST(s.index_time + interval '6 hours', s.outtime)
         THEN EXTRACT(EPOCH FROM (s.first_start - s.index_time)) / 3600.0
         ELSE NULL END AS lead_time_hours,
    LEAST(6.0, EXTRACT(EPOCH FROM (s.outtime - s.index_time)) / 3600.0) AS observed_horizon_hours,
    CASE WHEN s.outtime < s.index_time + interval '6 hours'
              AND (s.first_start IS NULL OR s.first_start >= s.outtime)
         THEN 1 ELSE 0 END AS exit_without_event,
    v.hr_last, v.hr_mean6, v.hr_min6, v.hr_max6,
    v.sbp_last, v.sbp_mean6, v.sbp_min6, v.sbp_max6,
    v.dbp_last, v.dbp_mean6, v.dbp_min6, v.dbp_max6,
    v.map_last, v.map_mean6, v.map_min6, v.map_max6,
    v.rr_last, v.rr_mean6, v.rr_min6, v.rr_max6,
    v.temp_last, v.temp_mean6, v.temp_min6, v.temp_max6,
    v.spo2_last, v.spo2_mean6, v.spo2_min6, v.spo2_max6,
    c.creatinine_last, c.sodium_last, c.potassium_last, c.bicarbonate_last,
    c.glucose_last, c.bun_last,
    b.lactate_last, b.ph_last,
    h.hemoglobin_last, h.hematocrit_last, h.wbc_last,
    c.creatinine_age_hours, c.sodium_age_hours, c.potassium_age_hours, c.bicarbonate_age_hours, c.glucose_age_hours, c.bun_age_hours, b.lactate_age_hours, b.ph_age_hours, h.hemoglobin_age_hours, h.hematocrit_age_hours, h.wbc_age_hours
FROM mimic_grid AS s
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.hr ORDER BY x.rel_hour DESC) FILTER (WHERE x.hr IS NOT NULL))[1] AS hr_last,
        AVG(x.hr) FILTER (WHERE x.hr IS NOT NULL) AS hr_mean6,
        MIN(x.hr) FILTER (WHERE x.hr IS NOT NULL) AS hr_min6,
        MAX(x.hr) FILTER (WHERE x.hr IS NOT NULL) AS hr_max6,
        (array_agg(x.sbp ORDER BY x.rel_hour DESC) FILTER (WHERE x.sbp IS NOT NULL))[1] AS sbp_last,
        AVG(x.sbp) FILTER (WHERE x.sbp IS NOT NULL) AS sbp_mean6,
        MIN(x.sbp) FILTER (WHERE x.sbp IS NOT NULL) AS sbp_min6,
        MAX(x.sbp) FILTER (WHERE x.sbp IS NOT NULL) AS sbp_max6,
        (array_agg(x.dbp ORDER BY x.rel_hour DESC) FILTER (WHERE x.dbp IS NOT NULL))[1] AS dbp_last,
        AVG(x.dbp) FILTER (WHERE x.dbp IS NOT NULL) AS dbp_mean6,
        MIN(x.dbp) FILTER (WHERE x.dbp IS NOT NULL) AS dbp_min6,
        MAX(x.dbp) FILTER (WHERE x.dbp IS NOT NULL) AS dbp_max6,
        (array_agg(x.map ORDER BY x.rel_hour DESC) FILTER (WHERE x.map IS NOT NULL))[1] AS map_last,
        AVG(x.map) FILTER (WHERE x.map IS NOT NULL) AS map_mean6,
        MIN(x.map) FILTER (WHERE x.map IS NOT NULL) AS map_min6,
        MAX(x.map) FILTER (WHERE x.map IS NOT NULL) AS map_max6,
        (array_agg(x.rr ORDER BY x.rel_hour DESC) FILTER (WHERE x.rr IS NOT NULL))[1] AS rr_last,
        AVG(x.rr) FILTER (WHERE x.rr IS NOT NULL) AS rr_mean6,
        MIN(x.rr) FILTER (WHERE x.rr IS NOT NULL) AS rr_min6,
        MAX(x.rr) FILTER (WHERE x.rr IS NOT NULL) AS rr_max6,
        (array_agg(x.temp ORDER BY x.rel_hour DESC) FILTER (WHERE x.temp IS NOT NULL))[1] AS temp_last,
        AVG(x.temp) FILTER (WHERE x.temp IS NOT NULL) AS temp_mean6,
        MIN(x.temp) FILTER (WHERE x.temp IS NOT NULL) AS temp_min6,
        MAX(x.temp) FILTER (WHERE x.temp IS NOT NULL) AS temp_max6,
        (array_agg(x.spo2 ORDER BY x.rel_hour DESC) FILTER (WHERE x.spo2 IS NOT NULL))[1] AS spo2_last,
        AVG(x.spo2) FILTER (WHERE x.spo2 IS NOT NULL) AS spo2_mean6,
        MIN(x.spo2) FILTER (WHERE x.spo2 IS NOT NULL) AS spo2_min6,
        MAX(x.spo2) FILTER (WHERE x.spo2 IS NOT NULL) AS spo2_max6
    FROM mimic_vitals AS x
    WHERE x.stay_id = s.stay_id
      AND x.rel_hour > s.index_hour - 6
      AND x.rel_hour <= s.index_hour
) AS v ON TRUE
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.creatinine ORDER BY x.rel_hour DESC) FILTER (WHERE x.creatinine IS NOT NULL))[1] AS creatinine_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.creatinine IS NOT NULL) AS creatinine_age_hours,
        (array_agg(x.sodium ORDER BY x.rel_hour DESC) FILTER (WHERE x.sodium IS NOT NULL))[1] AS sodium_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.sodium IS NOT NULL) AS sodium_age_hours,
        (array_agg(x.potassium ORDER BY x.rel_hour DESC) FILTER (WHERE x.potassium IS NOT NULL))[1] AS potassium_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.potassium IS NOT NULL) AS potassium_age_hours,
        (array_agg(x.bicarbonate ORDER BY x.rel_hour DESC) FILTER (WHERE x.bicarbonate IS NOT NULL))[1] AS bicarbonate_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.bicarbonate IS NOT NULL) AS bicarbonate_age_hours,
        (array_agg(x.glucose ORDER BY x.rel_hour DESC) FILTER (WHERE x.glucose IS NOT NULL))[1] AS glucose_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.glucose IS NOT NULL) AS glucose_age_hours,
        (array_agg(x.bun ORDER BY x.rel_hour DESC) FILTER (WHERE x.bun IS NOT NULL))[1] AS bun_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.bun IS NOT NULL) AS bun_age_hours
    FROM mimic_chem AS x
    WHERE x.stay_id = s.stay_id
      AND x.rel_hour > 0
      AND x.rel_hour <= s.index_hour
) AS c ON TRUE
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.lactate ORDER BY x.rel_hour DESC) FILTER (WHERE x.lactate IS NOT NULL))[1] AS lactate_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.lactate IS NOT NULL) AS lactate_age_hours,
        (array_agg(x.ph ORDER BY x.rel_hour DESC) FILTER (WHERE x.ph IS NOT NULL))[1] AS ph_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.ph IS NOT NULL) AS ph_age_hours
    FROM mimic_bg AS x
    WHERE x.stay_id = s.stay_id
      AND x.rel_hour > 0
      AND x.rel_hour <= s.index_hour
) AS b ON TRUE
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.hemoglobin ORDER BY x.rel_hour DESC) FILTER (WHERE x.hemoglobin IS NOT NULL))[1] AS hemoglobin_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.hemoglobin IS NOT NULL) AS hemoglobin_age_hours,
        (array_agg(x.hematocrit ORDER BY x.rel_hour DESC) FILTER (WHERE x.hematocrit IS NOT NULL))[1] AS hematocrit_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.hematocrit IS NOT NULL) AS hematocrit_age_hours,
        (array_agg(x.wbc ORDER BY x.rel_hour DESC) FILTER (WHERE x.wbc IS NOT NULL))[1] AS wbc_last,
        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.wbc IS NOT NULL) AS wbc_age_hours
    FROM mimic_cbc AS x
    WHERE x.stay_id = s.stay_id
      AND x.rel_hour > 0
      AND x.rel_hour <= s.index_hour
) AS h ON TRUE
ORDER BY s.stay_id, s.index_hour
) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');
