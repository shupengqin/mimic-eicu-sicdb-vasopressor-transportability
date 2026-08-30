# Data and code availability package

Prepared: 24 August 2026.

## Verified source datasets

| Dataset used | Verified release | Official citation and access | Access terms relevant to sharing |
| --- | --- | --- | --- |
| MIMIC-IV | v3.1, published 11 October 2024 | Johnson A et al. MIMIC-IV v3.1. PhysioNet. [DOI 10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58); [access page](https://physionet.org/content/mimiciv/3.1/) | Credentialed access, required training, and signed DUA. PhysioNet states that derived datasets or models should be treated as sensitive and, if shared, should use the same controlled terms. |
| eICU Collaborative Research Database | v2.0, published 15 April 2019 | Pollard T et al. eICU-CRD v2.0. PhysioNet. [DOI 10.13026/C2WM1R](https://doi.org/10.13026/C2WM1R); [access page](https://physionet.org/content/eicu-crd/2.0/) | Credentialed access, required training, and signed DUA under the PhysioNet Credentialed Health Data Licence 1.5.0. |
| Salzburg Intensive Care database | v1.0.8, published 10 September 2024 | Rodemund N et al. SICdb v1.0.8. PhysioNet. [DOI 10.13026/8m72-6j83](https://doi.org/10.13026/8m72-6j83); [access page](https://physionet.org/content/sicdb/1.0.8/) | Credentialed access, required training, signed DUA, and contributor review of the individual study under the Contributor Review Health Data Licence 1.5.0. |

Local database-scale checks were consistent with these releases: the MIMIC-IV installation contained 364,627 patients and 94,458 ICU stays; eICU-CRD contained 200,859 unit stays across 208 hospitals; and the SICdb local index explicitly identified release 1.0.8.

## Release classification

| Material | Public unrestricted repository? | Recommended handling |
| --- | --- | --- |
| MIMIC-IV/eICU/SICdb raw files and database tables | No | Readers obtain source data directly from PhysioNet under the applicable access process. |
| Patient-, stay-, or landmark-level extracted CSV files | No | Do not upload. Recreate locally from the source databases using the shared extraction code. |
| Row-level prediction NPZ/CSV files | No | Do not upload; these retain database-linked identifiers and derived patient-level information. |
| Joblib model objects | Not without written custodian approval | Treat as potentially sensitive derived models. Use an access-controlled PhysioNet project approved by the relevant custodians, or do not redistribute. |
| SQL extraction queries and Python analysis scripts | Yes, after scrubbing | Publish openly as required for reproducibility. Remove local paths, database passwords, usernames, and machine-specific settings first. |
| Environment/package manifest and file checksums | Yes | Publish openly. Add a lockfile or explicit package versions and operating-system notes. |
| Manuscript tables, figures, and publication-level aggregate source data | Usually, after disclosure review | Share only aggregates already suitable for publication. Review small cells, site identifiers, hospital-level outputs, and any database-specific disclosure risk. |
| Predictor dictionary, harmonization rules, and outcome mappings | Yes | Publish openly, provided no record-level examples or restricted values are included. |

## Ready-to-paste Data Availability statement

This study used deidentified, restricted-access data from MIMIC-IV version 3.1 (PhysioNet; doi:10.13026/kpb9-mt58), the eICU Collaborative Research Database version 2.0 (PhysioNet; doi:10.13026/C2WM1R), and the Salzburg Intensive Care database version 1.0.8 (PhysioNet; doi:10.13026/8m72-6j83). The authors are not permitted to redistribute the source databases or patient-, stay-, or landmark-level derived datasets. Qualified researchers may request MIMIC-IV and eICU-CRD access from PhysioNet after completing the required credentialing, training, and data-use agreements; SICdb additionally requires contributor review of the proposed study. The extraction and analysis code, predictor and outcome harmonization specifications, software environment information, and disclosure-reviewed aggregate source data supporting the published tables and figures are available at https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability. Frozen model objects and other potentially sensitive derived files will be shared only through an access-controlled mechanism approved by the relevant data custodians; if such approval is not obtained, they will not be redistributed.

## Ready-to-paste Code Availability statement

The SQL queries and Python code used for cohort construction, feature harmonization, model fitting, clustered inference, recalibration, sensitivity analyses, and figure generation are available at https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability. The repository includes exact package versions, random seeds, analysis checksums, a reproducible execution order, and disclosure-reviewed aggregate outputs for the reviewer-priority analyses. Machine-specific paths, credentials, source data, row-level extracts, row-level predictions, and model objects are not included.

## Minimum public repository contents

- `README` with access prerequisites, database versions, workflow order, expected aggregate outputs, and citation instructions.
- Extraction SQL for MIMIC-IV v3.1 and eICU-CRD v2.0.
- SICdb v1.0.8 extraction/harmonization code.
- Corrected modeling, clustered inference, recalibration, sensitivity, table, and figure scripts.
- Predictor dictionary, physiologic cleaning ranges, drug mappings, and outcome harmonization document.
- Environment file or lockfile, random seeds, and checksums from `corrected_run_manifest.json`.
- Disclosure-reviewed aggregate source data corresponding to every published table and figure.
- Completed TRIPOD+AI checklist and a versioned changelog.

## Materials that must be excluded from an unrestricted release

- `work/*.csv`, `work/*.npz`, extraction logs containing restricted values, and any database dump.
- Database usernames/passwords, local server details, and paths such as local drive locations.
- `corrected_*model.joblib` unless written approval and an access-controlled distribution route are in place.
- Hospital- or unit-level data with small cells or identifiers until disclosure review confirms publication is permitted.

## Chinese author confirmation notes

投稿前需要作者本人确认并补齐以下内容：

1. `[AUTHOR_INPUT_NEEDED]` 确认所有分析者对 MIMIC-IV v3.1、eICU-CRD v2.0 均具备有效 credential、培训和 DUA；确认 SICdb 本研究问题已经通过 contributor review。
2. `[AUTHOR_INPUT_NEEDED]` 提供最终公开代码仓库 URL、永久归档 DOI 和开源许可证；目前不能虚构这些地址。
3. `[AUTHOR_INPUT_NEEDED]` 在公开前删除脚本中的本机路径、PostgreSQL 用户名/密码和日志中的敏感内容。
4. `[AUTHOR_INPUT_NEEDED]` 向 PhysioNet/SICdb 数据管理方确认冻结模型是否可以通过受控项目共享。未经书面确认，不要把 joblib 模型放到公开 GitHub 或 Zenodo。
5. `[AUTHOR_INPUT_NEEDED]` 对论文级汇总表进行披露风险检查，尤其是 eICU 医院分层和 SICdb 单元分层结果。
