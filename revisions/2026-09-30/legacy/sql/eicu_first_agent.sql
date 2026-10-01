-- First documented continuous target vasopressor and concurrent first agents.
-- Drug-name normalization uses the same positive-rate definition as the main
-- eICU extraction and emits one aggregate row per unit stay.
\set ON_ERROR_STOP on
SET statement_timeout = 0;

COPY (
WITH agent_events AS (
    SELECT DISTINCT
        i.patientunitstayid,
        i.infusionoffset,
        CASE
            WHEN lower(i.drugname) ~ '(^|[^a-z])(norepinephrine|levophed|noradrenaline)([^a-z]|$)'
                THEN 'norepinephrine'
            WHEN lower(i.drugname) ~ '(^|[^a-z])(epinephrine|adrenaline)([^a-z]|$)'
                THEN 'epinephrine'
            WHEN lower(i.drugname) ~ '(^|[^a-z])phenylephrine([^a-z]|$)'
                THEN 'phenylephrine'
            WHEN lower(i.drugname) ~ '(^|[^a-z])vasopressin([^a-z]|$)'
                THEN 'vasopressin'
            WHEN lower(i.drugname) ~ '(^|[^a-z])dopamine([^a-z]|$)'
                THEN 'dopamine'
        END AS agent
    FROM infusiondrug AS i
    WHERE lower(i.drugname) ~ '(^|[^a-z])(norepinephrine|levophed|noradrenaline|vasopressin|phenylephrine|epinephrine|adrenaline|dopamine)([^a-z]|$)'
      AND (
          (btrim(i.drugrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.drugrate)::numeric > 0)
          OR (btrim(i.infusionrate) ~ '^[0-9]+([.][0-9]+)?$' AND btrim(i.infusionrate)::numeric > 0)
      )
), first_time AS (
    SELECT patientunitstayid, MIN(infusionoffset) AS first_start
    FROM agent_events
    WHERE agent IS NOT NULL
    GROUP BY patientunitstayid
), first_agents AS (
    SELECT
        e.patientunitstayid,
        f.first_start,
        string_agg(DISTINCT e.agent, '+' ORDER BY e.agent) AS first_agents,
        COUNT(DISTINCT e.agent)::int AS n_first_agents,
        bool_or(e.agent = 'norepinephrine') AS norepinephrine_at_first
    FROM agent_events AS e
    JOIN first_time AS f
      ON f.patientunitstayid = e.patientunitstayid
     AND f.first_start = e.infusionoffset
    GROUP BY e.patientunitstayid, f.first_start
)
SELECT
    p.patientunitstayid::bigint AS record_id,
    p.patienthealthsystemstayid::bigint AS patient_id,
    p.hospitalid::bigint AS hospital_id,
    f.first_start::double precision AS first_start_offset_minutes,
    f.first_start / 60.0 AS first_start_hour,
    f.first_agents,
    f.n_first_agents,
    f.norepinephrine_at_first::int AS norepinephrine_at_first
FROM first_agents AS f
JOIN patient AS p USING (patientunitstayid)
WHERE p.age = '> 89' OR (p.age ~ '^[0-9]+$' AND p.age::int >= 18)
ORDER BY p.patientunitstayid
) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');
