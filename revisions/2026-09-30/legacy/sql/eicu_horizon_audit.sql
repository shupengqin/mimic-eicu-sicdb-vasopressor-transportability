\set ON_ERROR_STOP on
SET statement_timeout = 0;

COPY (
WITH base AS (
    SELECT
        p.patientunitstayid,
        p.unitdischargeoffset
    FROM patient AS p
    WHERE (p.age = '> 89' OR (p.age ~ '^[0-9]+$' AND p.age::int >= 18))
      AND p.unitdischargeoffset >= 12 * 60
), pressor AS (
    SELECT
        i.patientunitstayid,
        MIN(i.infusionoffset) AS first_start
    FROM infusiondrug AS i
    WHERE lower(i.drugname) ~ '(^|[^a-z])(norepinephrine|levophed|noradrenaline|vasopressin|phenylephrine|epinephrine|adrenaline|dopamine)([^a-z]|$)'
      AND (
          (btrim(i.drugrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.drugrate)::numeric > 0)
          OR (btrim(i.infusionrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.infusionrate)::numeric > 0)
      )
    GROUP BY i.patientunitstayid
), candidate AS (
    SELECT
        b.patientunitstayid,
        g.index_hour,
        g.index_hour * 60 AS index_offset,
        b.unitdischargeoffset,
        p.first_start,
        b.unitdischargeoffset >= (g.index_hour + 6) * 60 AS complete_horizon
    FROM base AS b
    LEFT JOIN pressor AS p USING (patientunitstayid)
    CROSS JOIN LATERAL generate_series(6, 24) AS g(index_hour)
    WHERE b.unitdischargeoffset > g.index_hour * 60
      AND (p.first_start IS NULL OR p.first_start >= g.index_hour * 60)
)
SELECT
    'eicu'::text AS dataset,
    COUNT(*) AS at_risk_landmarks_present_at_index,
    COUNT(*) FILTER (WHERE complete_horizon) AS complete_six_hour_landmarks,
    COUNT(*) FILTER (WHERE NOT complete_horizon) AS incomplete_six_hour_landmarks,
    COUNT(*) FILTER (
        WHERE NOT complete_horizon
          AND first_start IS NOT NULL
          AND first_start < index_offset + 6 * 60
          AND first_start <= unitdischargeoffset
    ) AS incomplete_with_observed_event,
    COUNT(*) FILTER (
        WHERE NOT complete_horizon
          AND NOT (
              first_start IS NOT NULL
              AND first_start < index_offset + 6 * 60
              AND first_start <= unitdischargeoffset
          )
    ) AS incomplete_censored_without_observed_event,
    COUNT(DISTINCT patientunitstayid) AS at_risk_stays,
    COUNT(DISTINCT patientunitstayid) FILTER (WHERE complete_horizon) AS stays_with_complete_horizon_landmark,
    COUNT(DISTINCT patientunitstayid) FILTER (WHERE NOT complete_horizon) AS stays_with_incomplete_horizon_landmark
FROM candidate
) TO STDOUT WITH (FORMAT csv, HEADER true);
