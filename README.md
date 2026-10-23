# Core analysis code

This package contains only the modelling code retained in the revised manuscript. It covers sample construction, the perceived-environment model, the contextual model with six mapped indicators, prespecified sensitivity analyses, Holm adjustments, cluster-bootstrap indirect estimates, ordinal activity checks and the supplementary age-referenced visual-acuity model.

The repository contains code only. It does not include questionnaire responses, participant-to-unit links, mapped environmental data, private locations, spatial-data downloads, GIS preprocessing, figure production, legacy analyses or temporary audit scripts.

## Inputs

Three data files are required to run the analysis but are not included in this repository:

- `data/input_LOCAL_ONLY/geocoded_samples.xlsx`: the 242 questionnaire records using the variable names checked in `code/data_preparation.py`.
- `data/input_LOCAL_ONLY/sample_town_link_LOCAL_ONLY.csv`: `row_index` (0-241) and `town_osm_id` for matching respondents to aggregate unit indicators.
- `data/unit_environment_metrics.csv`: the six aggregate mapped indicators for the subdistrict/town units.

Access to the analysis data may be requested from the corresponding author under the conditions stated in the manuscript's Data Availability Statement.

## Run

Use Python 3.12 in a fresh environment:

```text
pip install -r requirements.txt
python -B code/run_analysis.py --components
python -B code/run_age_reference.py
```

Use `--resume` with `run_analysis.py` to retain model JSON files already completed in the current working copy. Outputs are written only under `results/`; logs and cache files remain within the package. The fixed random seed is `20260908`, and bootstrap analyses use 5,000 cluster resamples.

## Model scope

The continuous visual-acuity models use the 195-child primary sample: a reported examination within the previous 12 months and values for both eyes within 4.0-5.3. The main model contains 13 parent-perceived environmental indicators. The contextual model adds vegetation cover, NDVI, street/path density, potential activity-facility density, building coverage and OSM park-area share. The code also retains the sensitivity analyses specified in `code/data_preparation.py` and the supplementary Firth logistic model for the age-referenced binary outcome.

The code reproduces statistical associations and uncertainty estimates; it does not establish causal mediation.
