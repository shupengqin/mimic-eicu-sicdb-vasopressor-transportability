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
  AND p.unitdischargeoffset >= 12 * 60;
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
    g.index_hour,
    g.index_hour * 60 AS index_offset,
    p.first_start
FROM eicu_base AS b
LEFT JOIN eicu_pressor AS p USING (patientunitstayid)
CROSS JOIN LATERAL generate_series(6, 24) AS g(index_hour)
WHERE b.unitdischargeoffset >= (g.index_hour + 6) * 60
  AND (p.first_start IS NULL OR p.first_start >= g.index_hour * 60);
CREATE INDEX eicu_grid_stay_idx ON eicu_grid(patientunitstayid);
ANALYZE eicu_grid;

DROP TABLE IF EXISTS eicu_vitals;
CREATE TEMP TABLE eicu_vitals AS
SELECT
    v.patientunitstayid,
    v.observationoffset / 60.0 AS rel_hour,
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
CREATE INDEX eicu_vitals_stay_hour_idx ON eicu_vitals(patientunitstayid,rel_hour);
ANALYZE eicu_vitals;

DROP TABLE IF EXISTS eicu_labs;
CREATE TEMP TABLE eicu_labs AS
SELECT
    l.patientunitstayid,
    l.labresultoffset / 60.0 AS rel_hour,
    CASE WHEN lower(l.labname) LIKE '%lactate%' THEN l.labresult::double precision END AS lactate,
    CASE WHEN lower(l.labname) ~ '(hematocrit|hct)' THEN l.labresult::double precision END AS hematocrit,
    CASE WHEN lower(l.labname) ~ '(hemoglobin|hgb)' THEN l.labresult::double precision END AS hemoglobin,
    CASE WHEN lower(l.labname) ~ '(white blood|wbc)' THEN l.labresult::double precision END AS wbc,
    CASE WHEN lower(l.labname) LIKE '%creatinine%' THEN l.labresult::double precision END AS creatinine,
    CASE WHEN lower(l.labname) LIKE '%sodium%' THEN l.labresult::double precision END AS sodium,
    CASE WHEN lower(l.labname) LIKE '%potassium%' THEN l.labresult::double precision END AS potassium,
    CASE WHEN lower(l.labname) ~ '(bicarbonate|hco3|total co2)' THEN l.labresult::double precision END AS bicarbonate,
    CASE WHEN lower(l.labname) LIKE '%glucose%' THEN l.labresult::double precision END AS glucose,
    CASE WHEN lower(l.labname) ~ '(^|[^a-z])(bun|urea nitrogen)([^a-z]|$)' THEN l.labresult::double precision END AS bun,
    CASE WHEN lower(l.labname) LIKE '%ph%' THEN l.labresult::double precision END AS ph
FROM lab AS l
JOIN eicu_base AS b USING (patientunitstayid)
WHERE l.labresultoffset >= 0 AND l.labresultoffset <= 24 * 60
  AND l.labresult IS NOT NULL;
CREATE INDEX eicu_labs_stay_hour_idx ON eicu_labs(patientunitstayid,rel_hour);
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
    CASE WHEN s.first_start IS NOT NULL AND s.first_start < s.index_offset + 6 * 60 THEN 1 ELSE 0 END AS label,
    CASE WHEN s.first_start IS NULL THEN NULL ELSE (s.first_start - s.index_offset) / 60.0 END AS lead_time_hours,
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
