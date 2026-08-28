\set ON_ERROR_STOP on
COPY (
    SELECT
        patientunitstayid::bigint AS record_id,
        hospitalid::int AS hospital_id,
        unittype::text AS unit_name
    FROM patient
) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');
