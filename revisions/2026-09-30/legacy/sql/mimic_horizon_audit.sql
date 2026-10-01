\set ON_ERROR_STOP on
SET statement_timeout = 0;

COPY (
WITH base AS (
    SELECT
        i.stay_id,
        i.intime,
        i.outtime
    FROM mimiciv_icu.icustays AS i
    JOIN mimiciv_hosp.patients AS p USING (subject_id)
    WHERE (p.anchor_age + EXTRACT(YEAR FROM i.intime)::int - p.anchor_year) >= 18
      AND i.outtime >= i.intime + interval '12 hours'
), pressor AS (
    SELECT
        v.stay_id,
        MIN(v.starttime) AS first_start
    FROM mimiciv_derived.vasoactive_agent AS v
    WHERE COALESCE(v.norepinephrine, 0) > 0
       OR COALESCE(v.epinephrine, 0) > 0
       OR COALESCE(v.phenylephrine, 0) > 0
       OR COALESCE(v.vasopressin, 0) > 0
       OR COALESCE(v.dopamine, 0) > 0
    GROUP BY v.stay_id
), candidate AS (
    SELECT
        b.stay_id,
        g.index_hour,
        b.intime + g.index_hour * interval '1 hour' AS index_time,
        b.outtime,
        p.first_start,
        b.intime + (g.index_hour + 6) * interval '1 hour' <= b.outtime AS complete_horizon
    FROM base AS b
    LEFT JOIN pressor AS p USING (stay_id)
    CROSS JOIN LATERAL generate_series(6, 24) AS g(index_hour)
    WHERE b.intime + g.index_hour * interval '1 hour' < b.outtime
      AND (p.first_start IS NULL OR p.first_start >= b.intime + g.index_hour * interval '1 hour')
)
SELECT
    'mimiciv'::text AS dataset,
    COUNT(*) AS at_risk_landmarks_present_at_index,
    COUNT(*) FILTER (WHERE complete_horizon) AS complete_six_hour_landmarks,
    COUNT(*) FILTER (WHERE NOT complete_horizon) AS incomplete_six_hour_landmarks,
    COUNT(*) FILTER (
        WHERE NOT complete_horizon
          AND first_start IS NOT NULL
          AND first_start < index_time + interval '6 hours'
          AND first_start <= outtime
    ) AS incomplete_with_observed_event,
    COUNT(*) FILTER (
        WHERE NOT complete_horizon
          AND NOT (
              first_start IS NOT NULL
              AND first_start < index_time + interval '6 hours'
              AND first_start <= outtime
          )
    ) AS incomplete_censored_without_observed_event,
    COUNT(DISTINCT stay_id) AS at_risk_stays,
    COUNT(DISTINCT stay_id) FILTER (WHERE complete_horizon) AS stays_with_complete_horizon_landmark,
    COUNT(DISTINCT stay_id) FILTER (WHERE NOT complete_horizon) AS stays_with_incomplete_horizon_landmark
FROM candidate
) TO STDOUT WITH (FORMAT csv, HEADER true);
