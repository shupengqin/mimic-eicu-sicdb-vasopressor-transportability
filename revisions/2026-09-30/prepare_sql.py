"""Build isolated revision queries without changing the submitted code."""
import re
from config import *

RANGES = {'hr':(20,250),'sbp':(30,300),'dbp':(10,200),'map':(20,250),
          'rr':(2,100),'temp':(25,45),'spo2':(50,100),'creatinine':(.05,30),
          'sodium':(80,200),'potassium':(1,12),'bicarbonate':(5,60),
          'glucose':(20,1000),'bun':(1,200),'lactate':(.1,30),'ph':(6.5,8),
          'hemoglobin':(3,25),'hematocrit':(10,80),'wbc':(.1,300)}
LABS = list(RANGES)[7:]

def copy_query(query, filename):
    # psql client-side output, aggregate only for QC files.
    return "\\copy (" + query.replace('\n',' ') + ") TO 'results/" + filename + "' WITH CSV HEADER\n"

def qc(table, variables, dataset, suffix):
    rows=[]
    for v in variables:
        lo,hi=RANGES[v]
        rows.append(f"SELECT '{dataset}' AS dataset, '{v}' AS variable, count({v}) AS nonmissing_observations, count(*) FILTER (WHERE {v} IS NOT NULL AND NOT ({v} BETWEEN {lo} AND {hi})) AS rejected_observations FROM {table}")
    return copy_query(' UNION ALL '.join(rows),f'{dataset}_source_qc_{suffix}.csv')

for dataset in ('mimic','eicu'):
    s=(SOURCE/'sql'/f'prediction_time_{dataset}.sql').read_text(encoding='utf-8')
    # Temp names must not resolve to and drop any permanent user table.
    s=re.sub(r'DROP TABLE IF EXISTS (\w+);',r'DROP TABLE IF EXISTS pg_temp.\1;',s)
    if dataset=='mimic':
        s=s.replace('p.anchor_year_group::text AS time_group,','p.anchor_year_group::text AS time_group,\n        p.anchor_year,\n        p.dod,')
        s=s.replace('b.time_group,','b.time_group,\n    b.anchor_year,\n    b.dod,')
        s=s.replace('p.first_start >= b.intime','p.first_start > b.intime')
        # Capture counts before additional common range filters; apply before window summaries.
        for tab,variables in [('mimic_vitals',list(RANGES)[:7]),('mimic_chem',LABS[:6]),('mimic_bg',['lactate','ph']),('mimic_cbc',LABS[-3:])]:
            marker=f'CREATE INDEX {tab}_stay_hour_idx'
            insert=qc(tab,variables,dataset,tab)
            updates=', '.join(f'{v}=CASE WHEN {v} BETWEEN {RANGES[v][0]} AND {RANGES[v][1]} THEN {v} END' for v in variables)
            insert+=f'UPDATE {tab} SET {updates};\n'
            s=s.replace(marker,insert+marker)
        s=s.replace('s.time_group::text AS time_group,','s.time_group::text AS time_group,\n    s.time_year-s.anchor_year AS anchor_year_delta,\n    (s.dod IS NOT NULL AND s.dod=s.outtime::date)::int AS death_on_exit_date,')
        aliases={'creatinine':'c','sodium':'c','potassium':'c','bicarbonate':'c','glucose':'c','bun':'c','lactate':'b','ph':'b','hemoglobin':'h','hematocrit':'h','wbc':'h'}
    else:
        s=s.replace('p.unitdischargeoffset\nFROM patient','p.unitdischargeoffset, p.unitdischargestatus, p.unitdischargelocation\nFROM patient')
        s=s.replace('b.unit_name,\n    g.index_hour,','b.unit_name,\n    b.unitdischargeoffset, b.unitdischargestatus, b.unitdischargelocation,\n    g.index_hour,')
        s=s.replace('p.first_start >= g.index_hour','p.first_start > g.index_hour')
        s=s.replace('DROP TABLE IF EXISTS pg_temp.eicu_vitals;',qc('eicu_vitals_raw',list(RANGES)[:7],dataset,'vitals')+'DROP TABLE IF EXISTS pg_temp.eicu_vitals;')
        s=s.replace('DROP TABLE IF EXISTS pg_temp.eicu_labs;',qc('eicu_labs_raw',LABS,dataset,'labs')+'DROP TABLE IF EXISTS pg_temp.eicu_labs;')
        s=s.replace('NULL::int AS time_year,','NULL::int AS time_year,\n    s.hospitalid, s.unitdischargestatus, s.unitdischargelocation,')
        aliases={v:'l' for v in LABS}
    # Add source-measurement age without changing the selected laboratory value.
    for v in LABS:
        pattern=f'(array_agg(x.{v} ORDER BY x.rel_hour DESC) FILTER (WHERE x.{v} IS NOT NULL))[1] AS {v}_last'
        replacement=pattern+f',\n        s.index_hour - MAX(x.rel_hour) FILTER (WHERE x.{v} IS NOT NULL) AS {v}_age_hours'
        assert pattern in s,v
        s=s.replace(pattern,replacement)
    frm='FROM mimic_grid AS s' if dataset=='mimic' else 'FROM eicu_grid AS s'
    s=s.replace('\n'+frm,',\n    '+', '.join(f'{aliases[v]}.{v}_age_hours' for v in LABS)+'\n'+frm)
    (WORK/'sql'/f'{dataset}_revised.sql').write_text(s,encoding='utf-8')
print('Prepared isolated SQL with pre-aggregation QC, laboratory ages, and strict untreated-at-landmark boundary.')
