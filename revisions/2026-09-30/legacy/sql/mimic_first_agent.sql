-- First documented continuous target vasopressor and concurrent first agents.
-- This file emits one aggregate row per MIMIC-IV ICU stay and no patient-level
-- measurements beyond identifiers needed for local linkage.
\set ON_ERROR_STOP on
SET statement_timeout = 0;

COPY (
WITH agent_events AS (
    SELECT DISTINCT
        v.stay_id,
        v.starttime,
        a.agent
    FROM mimiciv_derived.vasoactive_agent AS v
    CROSS JOIN LATERAL (VALUES
        ('norepinephrine'::text, COALESCE(v.norepinephrine, 0)::double precision),
        ('epinephrine'::text, COALESCE(v.epinephrine, 0)::double precision),
        ('phenylephrine'::text, COALESCE(v.phenylephrine, 0)::double precision),
        ('vasopressin'::text, COALESCE(v.vasopressin, 0)::double precision),
        ('dopamine'::text, COALESCE(v.dopamine, 0)::double precision)
    ) AS a(agent, rate)
    WHERE a.rate > 0
), first_time AS (
    SELECT stay_id, MIN(starttime) AS first_start
    FROM agent_events
    GROUP BY stay_id
), first_agents AS (
    SELECT
        e.stay_id,
        f.first_start,
        string_agg(DISTINCT e.agent, '+' ORDER BY e.agent) AS first_agents,
        COUNT(DISTINCT e.agent)::int AS n_first_agents,
        bool_or(e.agent = 'norepinephrine') AS norepinephrine_at_first
    FROM agent_events AS e
    JOIN first_time AS f
      ON f.stay_id = e.stay_id
     AND f.first_start = e.starttime
    GROUP BY e.stay_id, f.first_start
)
SELECT
    i.stay_id::bigint AS record_id,
    i.subject_id::bigint AS patient_id,
    p.anchor_year_group::text AS time_group,
    f.first_start,
    EXTRACT(EPOCH FROM (f.first_start - i.intime)) / 3600.0 AS first_start_hour,
    f.first_agents,
    f.n_first_agents,
    f.norepinephrine_at_first::int AS norepinephrine_at_first
FROM first_agents AS f
JOIN mimiciv_icu.icustays AS i USING (stay_id)
JOIN mimiciv_hosp.patients AS p USING (subject_id)
WHERE (p.anchor_age + EXTRACT(YEAR FROM i.intime)::int - p.anchor_year) >= 18
ORDER BY i.stay_id
) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');
