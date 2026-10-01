-- Post hoc prediction-time-identifiable ICU-hour-6 estimand. Eligibility is
-- determined from information available at hour 6. Unit exit before hour 12
-- is treated as a competing outcome rather than as missing future follow-up.
\set ON_ERROR_STOP on
SET statement_timeout = 0;
SET lock_timeout = 0;
SET work_mem = '256MB';
SET temp_buffers = '128MB';

DROP TABLE IF EXISTS eicu_base;
CREATE TEMP TABLE eicu_base AS
SELECT
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.gender,
    CASE WHEN p.age = '> 89' THEN 90.0 WHEN p.age ~ '^[0-9]+$' THEN p.age::double precision ELSE NULL END AS age,
    p.hospitalid,
    p.unittype AS unit_name,
    p.unitdischargeoffset
FROM patient AS p
WHERE (p.age = '> 89' OR (p.age ~ '^[0-9]+$' AND p.age::int >= 18))
  AND p.unitdischargeoffset > 6 * 60;
CREATE INDEX eicu_base_stay_idx ON eicu_base(patientunitstayid);
ANALYZE eicu_base;

DROP TABLE IF EXISTS eicu_pressor;
CREATE TEMP TABLE eicu_pressor AS
SELECT
    i.patientunitstayid,
    MIN(i.infusionoffset) AS first_start
FROM infusiondrug AS i
WHERE lower(i.drugname) ~ '(^|[^a-z])(norepinephrine|levophed|noradrenaline|vasopressin|phenylephrine|epinephrine|adrenaline|dopamine)([^a-z]|$)'
  AND (
      (btrim(i.drugrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.drugrate)::numeric > 0)
      OR (btrim(i.infusionrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.infusionrate)::numeric > 0)
  )
GROUP BY i.patientunitstayid;
CREATE INDEX eicu_pressor_stay_idx ON eicu_pressor(patientunitstayid);
ANALYZE eicu_pressor;

DROP TABLE IF EXISTS eicu_grid;
CREATE TEMP TABLE eicu_grid AS
SELECT
    b.patientunitstayid,
    b.patienthealthsystemstayid,
    b.gender,
    b.age,
    b.hospitalid,
    b.unit_name,
    b.unitdischargeoffset,
    g.index_hour,
    g.index_hour * 60 AS index_offset,
    p.first_start
FROM eicu_base AS b
LEFT JOIN eicu_pressor AS p USING (patientunitstayid)
CROSS JOIN LATERAL (VALUES (6)) AS g(index_hour)
WHERE p.first_start IS NULL OR p.first_start >= g.index_hour * 60;
CREATE INDEX eicu_grid_stay_idx ON eicu_grid(patientunitstayid);
ANALYZE eicu_grid;

-- Keep one value per stay, observation time and variable. Periodic vitals are
-- preferred over aperiodic values, and nurse-charted temperature is the final
-- fallback. Range checks occur before window aggregation so outliers cannot
-- distort means, minima or maxima.
DROP TABLE IF EXISTS eicu_vitals_raw;
CREATE TEMP TABLE eicu_vitals_raw AS
SELECT
    v.patientunitstayid,
    v.observationoffset / 60.0 AS rel_hour,
    1::int AS source_priority,
    v.vitalperiodicid::bigint AS source_id,
    v.heartrate::double precision AS hr,
    v.systemicsystolic::double precision AS sbp,
    v.systemicdiastolic::double precision AS dbp,
    v.systemicmean::double precision AS map,
    v.respiration::double precision AS rr,
    v.temperature::double precision AS temp,
    v.sao2::double precision AS spo2
FROM vitalperiodic AS v
JOIN eicu_base AS b USING (patientunitstayid)
WHERE v.observationoffset >= 0 AND v.observationoffset <= 24 * 60
UNION ALL
SELECT
    v.patientunitstayid,
    v.observationoffset / 60.0 AS rel_hour,
    2::int AS source_priority,
    v.vitalaperiodicid::bigint AS source_id,
    NULL::double precision AS hr,
    v.noninvasivesystolic::double precision AS sbp,
    v.noninvasivediastolic::double precision AS dbp,
    v.noninvasivemean::double precision AS map,
    NULL::double precision AS rr,
    NULL::double precision AS temp,
    NULL::double precision AS spo2
FROM vitalaperiodic AS v
JOIN eicu_base AS b USING (patientunitstayid)
WHERE v.observationoffset >= 0 AND v.observationoffset <= 24 * 60
UNION ALL
SELECT
    n.patientunitstayid,
    n.nursingchartoffset / 60.0 AS rel_hour,
    3::int AS source_priority,
    n.nursingchartid::bigint AS source_id,
    NULL::double precision AS hr,
    NULL::double precision AS sbp,
    NULL::double precision AS dbp,
    NULL::double precision AS map,
    NULL::double precision AS rr,
    CASE
        WHEN lower(n.nursingchartcelltypevalname) = 'temperature (c)'
            THEN btrim(n.nursingchartvalue)::double precision
        WHEN lower(n.nursingchartcelltypevalname) = 'temperature (f)'
            THEN (btrim(n.nursingchartvalue)::double precision - 32.0) * 5.0 / 9.0
        ELSE NULL
    END AS temp,
    NULL::double precision AS spo2
FROM nursecharting AS n
JOIN eicu_base AS b USING (patientunitstayid)
WHERE n.nursingchartoffset >= 0
  AND n.nursingchartoffset <= 24 * 60
  AND lower(n.nursingchartcelltypevalname) IN ('temperature (c)', 'temperature (f)')
  AND btrim(n.nursingchartvalue) ~ '^[0-9]+([.][0-9]+)?$';

DROP TABLE IF EXISTS eicu_vitals;
CREATE TEMP TABLE eicu_vitals AS
WITH long_values AS (
    SELECT r.patientunitstayid, r.rel_hour, r.source_priority, r.source_id,
           u.variable, u.value
    FROM eicu_vitals_raw AS r
    CROSS JOIN LATERAL (VALUES
        ('hr'::text, CASE WHEN r.hr BETWEEN 20 AND 250 THEN r.hr END),
        ('sbp'::text, CASE WHEN r.sbp BETWEEN 30 AND 300 THEN r.sbp END),
        ('dbp'::text, CASE WHEN r.dbp BETWEEN 10 AND 200 THEN r.dbp END),
        ('map'::text, CASE WHEN r.map BETWEEN 20 AND 250 THEN r.map END),
        ('rr'::text, CASE WHEN r.rr BETWEEN 2 AND 100 THEN r.rr END),
        ('temp'::text, CASE WHEN r.temp BETWEEN 25 AND 45 THEN r.temp END),
        ('spo2'::text, CASE WHEN r.spo2 BETWEEN 50 AND 100 THEN r.spo2 END)
    ) AS u(variable, value)
    WHERE u.value IS NOT NULL
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY patientunitstayid, rel_hour, variable
        ORDER BY source_priority, source_id DESC
    ) AS rn
    FROM long_values
)
SELECT
    patientunitstayid,
    rel_hour,
    MAX(value) FILTER (WHERE variable = 'hr') AS hr,
    MAX(value) FILTER (WHERE variable = 'sbp') AS sbp,
    MAX(value) FILTER (WHERE variable = 'dbp') AS dbp,
    MAX(value) FILTER (WHERE variable = 'map') AS map,
    MAX(value) FILTER (WHERE variable = 'rr') AS rr,
    MAX(value) FILTER (WHERE variable = 'temp') AS temp,
    MAX(value) FILTER (WHERE variable = 'spo2') AS spo2
FROM ranked
WHERE rn = 1
GROUP BY patientunitstayid, rel_hour;
CREATE INDEX eicu_vitals_stay_hour_idx ON eicu_vitals(patientunitstayid, rel_hour);
ANALYZE eicu_vitals;

-- Classify laboratory names with token boundaries. This prevents phosphate
-- from being treated as pH and lactate dehydrogenase from being treated as
-- lactate. The latest revised result is selected deterministically below.
DROP TABLE IF EXISTS eicu_labs_raw;
CREATE TEMP TABLE eicu_labs_raw AS
SELECT
    l.patientunitstayid,
    l.labresultoffset / 60.0 AS rel_hour,
    l.labid::bigint AS source_id,
    l.labresultrevisedoffset::double precision AS revised_offset,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(lactate|lactic acid)([^a-z]|$)'
              AND lower(l.labname) !~ '(dehydrogenase|(^|[^a-z])ldh([^a-z]|$))'
         THEN l.labresult::double precision END AS lactate,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(hematocrit|hct)([^a-z]|$)'
         THEN l.labresult::double precision END AS hematocrit,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(hemoglobin|hgb)([^a-z]|$)'
         THEN l.labresult::double precision END AS hemoglobin,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(white blood cell|wbc)([^a-z]|$)'
         THEN l.labresult::double precision END AS wbc,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])creatinine([^a-z]|$)'
         THEN l.labresult::double precision END AS creatinine,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])sodium([^a-z]|$)'
         THEN l.labresult::double precision END AS sodium,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])potassium([^a-z]|$)'
         THEN l.labresult::double precision END AS potassium,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(bicarbonate|hco3|total co2)([^a-z]|$)'
         THEN l.labresult::double precision END AS bicarbonate,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])glucose([^a-z]|$)'
         THEN l.labresult::double precision END AS glucose,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(bun|urea nitrogen)([^a-z]|$)'
         THEN l.labresult::double precision END AS bun,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])ph([^a-z]|$)'
         THEN l.labresult::double precision END AS ph
FROM lab AS l
JOIN eicu_base AS b USING (patientunitstayid)
WHERE l.labresultoffset >= 0 AND l.labresultoffset <= 24 * 60
  AND l.labresult IS NOT NULL;

DROP TABLE IF EXISTS eicu_labs;
CREATE TEMP TABLE eicu_labs AS
WITH long_values AS (
    SELECT r.patientunitstayid, r.rel_hour, r.source_id, r.revised_offset,
           u.variable, u.value
    FROM eicu_labs_raw AS r
    CROSS JOIN LATERAL (VALUES
        ('lactate'::text, CASE WHEN r.lactate BETWEEN 0.1 AND 30 THEN r.lactate END),
        ('hematocrit'::text, CASE WHEN r.hematocrit BETWEEN 10 AND 80 THEN r.hematocrit END),
        ('hemoglobin'::text, CASE WHEN r.hemoglobin BETWEEN 3 AND 25 THEN r.hemoglobin END),
        ('wbc'::text, CASE WHEN r.wbc BETWEEN 0.1 AND 300 THEN r.wbc END),
        ('creatinine'::text, CASE WHEN r.creatinine BETWEEN 0.05 AND 30 THEN r.creatinine END),
        ('sodium'::text, CASE WHEN r.sodium BETWEEN 80 AND 200 THEN r.sodium END),
        ('potassium'::text, CASE WHEN r.potassium BETWEEN 1 AND 12 THEN r.potassium END),
        ('bicarbonate'::text, CASE WHEN r.bicarbonate BETWEEN 5 AND 60 THEN r.bicarbonate END),
        ('glucose'::text, CASE WHEN r.glucose BETWEEN 20 AND 1000 THEN r.glucose END),
        ('bun'::text, CASE WHEN r.bun BETWEEN 1 AND 200 THEN r.bun END),
        ('ph'::text, CASE WHEN r.ph BETWEEN 6.5 AND 8 THEN r.ph END)
    ) AS u(variable, value)
    WHERE u.value IS NOT NULL
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY patientunitstayid, rel_hour, variable
        ORDER BY revised_offset DESC NULLS LAST, source_id DESC
    ) AS rn
    FROM long_values
)
SELECT
    patientunitstayid,
    rel_hour,
    MAX(value) FILTER (WHERE variable = 'lactate') AS lactate,
    MAX(value) FILTER (WHERE variable = 'hematocrit') AS hematocrit,
    MAX(value) FILTER (WHERE variable = 'hemoglobin') AS hemoglobin,
    MAX(value) FILTER (WHERE variable = 'wbc') AS wbc,
    MAX(value) FILTER (WHERE variable = 'creatinine') AS creatinine,
    MAX(value) FILTER (WHERE variable = 'sodium') AS sodium,
    MAX(value) FILTER (WHERE variable = 'potassium') AS potassium,
    MAX(value) FILTER (WHERE variable = 'bicarbonate') AS bicarbonate,
    MAX(value) FILTER (WHERE variable = 'glucose') AS glucose,
    MAX(value) FILTER (WHERE variable = 'bun') AS bun,
    MAX(value) FILTER (WHERE variable = 'ph') AS ph
FROM ranked
WHERE rn = 1
GROUP BY patientunitstayid, rel_hour;
CREATE INDEX eicu_labs_stay_hour_idx ON eicu_labs(patientunitstayid, rel_hour);
ANALYZE eicu_labs;

COPY (
SELECT
    'eicu'::text AS dataset,
    s.patientunitstayid::bigint AS record_id,
    s.patienthealthsystemstayid::bigint AS patient_id,
    s.unit_name::text AS unit_name,
    NULL::int AS time_year,
    s.index_hour::int AS index_hour,
    s.age::double precision AS age,
    CASE WHEN lower(btrim(s.gender)) LIKE 'm%' THEN 1.0 WHEN lower(btrim(s.gender)) LIKE 'f%' THEN 0.0 ELSE NULL END AS sex_male,
    CASE WHEN s.first_start IS NOT NULL
              AND s.first_start < LEAST(s.index_offset + 6 * 60, s.unitdischargeoffset)
         THEN 1 ELSE 0 END AS label,
    CASE WHEN s.first_start IS NOT NULL
              AND s.first_start < LEAST(s.index_offset + 6 * 60, s.unitdischargeoffset)
         THEN (s.first_start - s.index_offset) / 60.0
         ELSE NULL END AS lead_time_hours,
    LEAST(6.0, (s.unitdischargeoffset - s.index_offset) / 60.0) AS observed_horizon_hours,
    CASE WHEN s.unitdischargeoffset < s.index_offset + 6 * 60
              AND (s.first_start IS NULL OR s.first_start >= s.unitdischargeoffset)
         THEN 1 ELSE 0 END AS competing_exit,
    v.hr_last, v.hr_mean6, v.hr_min6, v.hr_max6,
    v.sbp_last, v.sbp_mean6, v.sbp_min6, v.sbp_max6,
    v.dbp_last, v.dbp_mean6, v.dbp_min6, v.dbp_max6,
    v.map_last, v.map_mean6, v.map_min6, v.map_max6,
    v.rr_last, v.rr_mean6, v.rr_min6, v.rr_max6,
    v.temp_last, v.temp_mean6, v.temp_min6, v.temp_max6,
    v.spo2_last, v.spo2_mean6, v.spo2_min6, v.spo2_max6,
    l.creatinine_last, l.sodium_last, l.potassium_last, l.bicarbonate_last,
    l.glucose_last, l.bun_last, l.lactate_last, l.ph_last,
    l.hemoglobin_last, l.hematocrit_last, l.wbc_last
FROM eicu_grid AS s
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.hr ORDER BY x.rel_hour DESC) FILTER (WHERE x.hr IS NOT NULL))[1] AS hr_last,
        AVG(x.hr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.hr IS NOT NULL) AS hr_mean6,
        MIN(x.hr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.hr IS NOT NULL) AS hr_min6,
        MAX(x.hr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.hr IS NOT NULL) AS hr_max6,
        (array_agg(x.sbp ORDER BY x.rel_hour DESC) FILTER (WHERE x.sbp IS NOT NULL))[1] AS sbp_last,
        AVG(x.sbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.sbp IS NOT NULL) AS sbp_mean6,
        MIN(x.sbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.sbp IS NOT NULL) AS sbp_min6,
        MAX(x.sbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.sbp IS NOT NULL) AS sbp_max6,
        (array_agg(x.dbp ORDER BY x.rel_hour DESC) FILTER (WHERE x.dbp IS NOT NULL))[1] AS dbp_last,
        AVG(x.dbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.dbp IS NOT NULL) AS dbp_mean6,
        MIN(x.dbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.dbp IS NOT NULL) AS dbp_min6,
        MAX(x.dbp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.dbp IS NOT NULL) AS dbp_max6,
        (array_agg(x.map ORDER BY x.rel_hour DESC) FILTER (WHERE x.map IS NOT NULL))[1] AS map_last,
        AVG(x.map) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.map IS NOT NULL) AS map_mean6,
        MIN(x.map) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.map IS NOT NULL) AS map_min6,
        MAX(x.map) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.map IS NOT NULL) AS map_max6,
        (array_agg(x.rr ORDER BY x.rel_hour DESC) FILTER (WHERE x.rr IS NOT NULL))[1] AS rr_last,
        AVG(x.rr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.rr IS NOT NULL) AS rr_mean6,
        MIN(x.rr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.rr IS NOT NULL) AS rr_min6,
        MAX(x.rr) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.rr IS NOT NULL) AS rr_max6,
        (array_agg(x.temp ORDER BY x.rel_hour DESC) FILTER (WHERE x.temp IS NOT NULL))[1] AS temp_last,
        AVG(x.temp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.temp IS NOT NULL) AS temp_mean6,
        MIN(x.temp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.temp IS NOT NULL) AS temp_min6,
        MAX(x.temp) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.temp IS NOT NULL) AS temp_max6,
        (array_agg(x.spo2 ORDER BY x.rel_hour DESC) FILTER (WHERE x.spo2 IS NOT NULL))[1] AS spo2_last,
        AVG(x.spo2) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.spo2 IS NOT NULL) AS spo2_mean6,
        MIN(x.spo2) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.spo2 IS NOT NULL) AS spo2_min6,
        MAX(x.spo2) FILTER (WHERE x.rel_hour > s.index_hour - 6 AND x.rel_hour <= s.index_hour AND x.spo2 IS NOT NULL) AS spo2_max6
    FROM eicu_vitals AS x
    WHERE x.patientunitstayid = s.patientunitstayid
      AND x.rel_hour > s.index_hour - 6
      AND x.rel_hour <= s.index_hour
) AS v ON TRUE
LEFT JOIN LATERAL (
    SELECT
        (array_agg(x.creatinine ORDER BY x.rel_hour DESC) FILTER (WHERE x.creatinine IS NOT NULL))[1] AS creatinine_last,
        (array_agg(x.sodium ORDER BY x.rel_hour DESC) FILTER (WHERE x.sodium IS NOT NULL))[1] AS sodium_last,
        (array_agg(x.potassium ORDER BY x.rel_hour DESC) FILTER (WHERE x.potassium IS NOT NULL))[1] AS potassium_last,
        (array_agg(x.bicarbonate ORDER BY x.rel_hour DESC) FILTER (WHERE x.bicarbonate IS NOT NULL))[1] AS bicarbonate_last,
        (array_agg(x.glucose ORDER BY x.rel_hour DESC) FILTER (WHERE x.glucose IS NOT NULL))[1] AS glucose_last,
        (array_agg(x.bun ORDER BY x.rel_hour DESC) FILTER (WHERE x.bun IS NOT NULL))[1] AS bun_last,
        (array_agg(x.lactate ORDER BY x.rel_hour DESC) FILTER (WHERE x.lactate IS NOT NULL))[1] AS lactate_last,
        (array_agg(x.ph ORDER BY x.rel_hour DESC) FILTER (WHERE x.ph IS NOT NULL))[1] AS ph_last,
        (array_agg(x.hemoglobin ORDER BY x.rel_hour DESC) FILTER (WHERE x.hemoglobin IS NOT NULL))[1] AS hemoglobin_last,
        (array_agg(x.hematocrit ORDER BY x.rel_hour DESC) FILTER (WHERE x.hematocrit IS NOT NULL))[1] AS hematocrit_last,
        (array_agg(x.wbc ORDER BY x.rel_hour DESC) FILTER (WHERE x.wbc IS NOT NULL))[1] AS wbc_last
    FROM eicu_labs AS x
    WHERE x.patientunitstayid = s.patientunitstayid
      AND x.rel_hour > 0
      AND x.rel_hour <= s.index_hour
) AS l ON TRUE
ORDER BY s.patientunitstayid, s.index_hour
) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');
