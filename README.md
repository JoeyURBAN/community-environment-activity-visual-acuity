# Core analysis code

This package contains the core modelling code for the revised manuscript. It covers sample construction, the perceived-environment model, the contextual model with five mapped indicators, retained sensitivity analyses, Holm adjustments, cluster-bootstrap indirect estimates, ordinal activity checks and the supplementary age-referenced visual-acuity model. The original six-indicator contextual model and an NDVI-substitution model are retained as supplementary comparisons.

The repository contains code only. It does not include questionnaire responses, participant-to-unit links, mapped environmental data, private locations, spatial-data downloads, GIS preprocessing, figure production, legacy analyses or temporary audit scripts.

## Inputs

Three data files are required to run the analysis but are not included in this repository:

- `data/input_LOCAL_ONLY/geocoded_samples.xlsx`: the 242 questionnaire records using the variable names checked in `code/data_preparation.py`.
- `data/input_LOCAL_ONLY/sample_town_link_LOCAL_ONLY.csv`: `row_index` (0-241) and `town_osm_id` for matching respondents to aggregate unit indicators.
- `data/unit_environment_metrics.csv`: the six aggregate mapped indicators for the subdistrict/town units, with columns `town_osm_id`, `Vegetation`, `NDVI`, `Road`, `Facility`, `Building` and `ParkShare`. All six source measures are loaded for consistency with the supplementary comparisons; the main contextual equation uses only five.

Access to the analysis data may be requested from the corresponding author under the conditions stated in the manuscript's Data Availability Statement.

## Run

Use Python 3.12 in a fresh environment:

```text
pip install -r requirements.txt
python -B code/run_analysis.py --components
python -B code/run_age_reference.py
```

To include the two supplementary greenness comparisons, use:

```text
python -B code/run_analysis.py --components --greenness-comparisons
```

Use a fresh working copy when changing from the previous six-indicator default. `--resume` retains completed model JSON files only when the model name, sample size, unit count, predictors, covariates and coding match. It rejects an old six-indicator `contextual.json`; it does not check input-file identity, so input data must remain unchanged. Preserve earlier outputs separately. Outputs are written only under `results/`; logs and cache files remain within the package. The fixed random seed is `20260908`, and bootstrap analyses use 5,000 cluster resamples.

## Model scope

The main continuous visual-acuity analyses use the 195-child sample: a reported examination within the previous 12 months and values for both eyes within 4.0-5.3. The primary model contains 13 parent-perceived environmental indicators. The main contextual model adds vegetation cover, street/path density, potential activity-facility density, building coverage and OSM park-area share. NDVI is omitted following the reviewer-requested collinearity assessment, not selection by statistical significance.

| Output name | Role | Mapped predictors |
| --- | --- | --- |
| `contextual` | Main contextual model | Vegetation, Road, Facility, Building, ParkShare |
| `joint_original` | Supplementary original joint model | Vegetation, NDVI, Road, Facility, Building, ParkShare |
| `ndvi_only` | Supplementary NDVI-substitution model | NDVI, Road, Facility, Building, ParkShare |

All three specifications use the same 195 children, parental ratings and child/family covariates. The four paired contextual sensitivity models also use the five-indicator specification. Each equation reports predictor VIFs without automatic selection. Holm families are calculated separately for each specification: the five-indicator models have 38 activity-path tests, 18 direct environmental-acuity tests and 36 specific indirect tests; the six-indicator comparison has 40, 19 and 38, respectively. The separate domain family remains 14 tests.

The code also retains the sensitivity analyses specified in `code/data_preparation.py` and the supplementary Firth logistic model for the age-referenced binary outcome. This is a core-code package, not an export of every supplementary audit: the separately reported single-record exclusion and leave-one-out analyses are not added by this update.

The code reproduces statistical associations and uncertainty estimates; it does not establish causal mediation.

## Configuration checks

The tests below use synthetic records only and do not run the study models:

```text
python -B -m unittest discover -s tests -v
```
