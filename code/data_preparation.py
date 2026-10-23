"""Construct the analysis data and retained model specifications.

The child-level survey and its unit-link file are restricted inputs and are
not distributed with this package. Only aggregate, non-participant outputs
are written by this module.
"""

import numpy as np
import pandas as pd

from config import COVARIATES, DOMAINS, INPUTS, ITEMS, OBJECTIVE, SUBJECTIVE
from runtime import save_csv, save_json


REQUIRED_SURVEY_COLUMNS = [
    'child_gender', 'child_age',
    'father_diopter_L', 'father_diopter_R',
    'mother_diopter_L', 'mother_diopter_R',
    'father_edu', 'mother_edu', 'income',
    'last_vision_exam', 'child_vision_L', 'child_vision_R',
    'outdoor_daily', 'screen_time', 'near_work_time',
    'built_density_pop', 'built_density_bldg', 'safety_overall',
    'walk2park_min',
] + [row[1] for row in ITEMS]


def load_data(export=True):
    raw = pd.read_excel(INPUTS['survey'])
    missing_columns = sorted(set(REQUIRED_SURVEY_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f'Missing required survey columns: {missing_columns}')
    if len(raw) != 242:
        raise ValueError(f'Expected 242 survey records, found {len(raw)}')

    parental_myopia = (
        (raw[['father_diopter_L', 'father_diopter_R']].max(axis=1) > 0).astype(int)
        + (raw[['mother_diopter_L', 'mother_diopter_R']].max(axis=1) > 0).astype(int)
    )

    data = pd.DataFrame({
        'row_id': np.arange(len(raw)),
        'Outdoor': raw.outdoor_daily.astype(float),
        'Screen': raw.screen_time.astype(float),
        'VA': raw[['child_vision_L', 'child_vision_R']].mean(axis=1).round(2),
        'VA_Lower': raw[['child_vision_L', 'child_vision_R']].min(axis=1).round(2),
        'VA_Left': raw.child_vision_L.astype(float),
        'VA_Right': raw.child_vision_R.astype(float),
        'Age': raw.child_age.astype(float) + 2,
        'Female': (raw.child_gender == 2).astype(int),
        'ParentsMyopic': parental_myopia,
        'ParentOne': (parental_myopia == 1).astype(int),
        'ParentTwo': (parental_myopia == 2).astype(int),
        'Income': raw.income.astype(float),
        'Education': raw[['father_edu', 'mother_edu']].mean(axis=1),
        'NearWork': raw.near_work_time.astype(float),
        'Exam': raw.last_vision_exam.astype(float),
        'DensityPop': raw.built_density_pop.astype(float),
        'DensityBuilt': raw.built_density_bldg.astype(float),
        'OverallSafety': raw.safety_overall.astype(float),
        'ParkTimeOriginal': raw.walk2park_min.astype(float),
    })

    for name, column, _, _, _ in ITEMS:
        values = raw[column].astype(float)
        if not values.between(1, 5).all():
            raise ValueError(f'{column} contains values outside 1-5')
        data[name] = 6 - values if name == 'ParkAccess' else values

    data['RangeEligible'] = (
        data.VA_Left.between(4.0, 5.3) & data.VA_Right.between(4.0, 5.3)
    )
    data['RecentEligible'] = data.Exam.isin([1, 2, 3])
    data['HasExam'] = data.Exam != 6
    data['PrimaryEligible'] = data.HasExam & data.RangeEligible & data.RecentEligible
    data['AgeReference'] = data.Age.map({3: 4.7, 4: 4.8, 5: 4.8, 6: 4.9})
    data['BelowAgeReference'] = (
        data[['VA_Left', 'VA_Right']].min(axis=1) < data.AgeReference
    ).astype(int)

    links = pd.read_csv(INPUTS['unit_links']).sort_values('row_index')
    if links.row_index.tolist() != list(range(len(raw))):
        raise ValueError('Unit-link rows do not align with the survey rows')
    if links.town_osm_id.isna().any():
        raise ValueError('Missing administrative-unit identifiers')

    metrics = pd.read_csv(INPUTS['unit_metrics'])
    if metrics.town_osm_id.duplicated().any():
        raise ValueError('Unit metrics contain duplicate identifiers')
    joined = links[['row_index', 'town_osm_id']].merge(
        metrics, on='town_osm_id', how='left', validate='many_to_one'
    ).sort_values('row_index')
    if joined[OBJECTIVE].isna().any().any():
        raise ValueError('At least one mapped indicator is missing after unit matching')

    data['town_osm_id'] = joined.town_osm_id.to_numpy()
    data['group'] = pd.factorize(data.town_osm_id, sort=False)[0]
    for name in OBJECTIVE:
        data[name + '_raw'] = joined[name].to_numpy(float)

    scaling = {}
    for name in OBJECTIVE:
        values = data[name + '_raw']
        if (values < 0).any() and name != 'NDVI':
            raise ValueError(f'{name} contains negative values')
        if name in ['Vegetation', 'Building', 'ParkShare'] and (values > 1 + 1e-8).any():
            raise ValueError(f'{name} contains proportions above 1')
        transformed = np.log1p(values) if name in ['Road', 'Facility'] else values
        reference = transformed[data.PrimaryEligible]
        mean = reference.mean()
        sd = reference.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f'{name} has no usable variation')
        data[name] = (transformed - mean) / sd
        scaling[name] = {
            'mean': mean,
            'sd': sd,
            'n': len(reference),
            'transform': 'log1p' if name in ['Road', 'Facility'] else 'identity',
            'weighting': '195 children; repeated unit values weighted by sampled children',
        }

    primary = data[data.PrimaryEligible].copy()
    counts = (
        int(data.HasExam.sum()),
        int((data.HasExam & data.RecentEligible).sum()),
        int((data.HasExam & data.RangeEligible).sum()),
        len(primary),
        primary.group.nunique(),
    )
    if counts != (238, 204, 226, 195, 90):
        raise ValueError(f'Unexpected sample counts: {counts}')
    required = SUBJECTIVE + COVARIATES + OBJECTIVE + ['Outdoor', 'Screen', 'VA']
    if primary[required].isna().any().any():
        raise ValueError('Unexpected missing values in the primary analysis data')

    cluster_sizes = primary.groupby('group').size()
    cluster_summary = (
        int(cluster_sizes[cluster_sizes >= 2].sum()),
        int((cluster_sizes >= 2).sum()),
        int((cluster_sizes == 1).sum()),
    )
    if cluster_summary != (149, 44, 46):
        raise ValueError(f'Unexpected cluster-size distribution: {cluster_summary}')
    if int(primary.BelowAgeReference.sum()) != 21:
        raise ValueError('Unexpected age-reference event count')

    if export:
        save_json('audit/data_audit.json', {
            'counts': {
                'responses': 242,
                'previous_examination': 238,
                'recent_examination': 204,
                'acuity_range': 226,
                'primary': 195,
                'primary_units': 90,
                'below_age_reference': 21,
            },
            'scaling': scaling,
            'age_reference_note': (
                'Current survey age proxies age at examination; this is not a diagnosis.'
            ),
        })
        flow = []
        for label, mask in [
            ('Original responses', np.ones(len(data), dtype=bool)),
            ('Previous examination', data.HasExam),
            ('Within 12 months', data.HasExam & data.RecentEligible),
            ('Within 12 months and both eyes 4.0-5.3', data.PrimaryEligible),
        ]:
            flow.append({
                'stage': label,
                'n': int(mask.sum()),
                'units': int(data.loc[mask, 'group'].nunique()),
            })
        save_csv('tables/sample_flow.csv', flow)
        save_csv(
            'tables/descriptives_primary.csv',
            primary[SUBJECTIVE + COVARIATES + OBJECTIVE
                    + ['Outdoor', 'Screen', 'VA', 'VA_Lower']]
            .describe().T.reset_index(names='variable'),
        )
        save_csv(
            'tables/age_reference_counts.csv',
            primary.groupby('Age').agg(
                n=('row_id', 'size'),
                below=('BelowAgeReference', 'sum'),
                reference=('AgeReference', 'first'),
            ).reset_index(),
        )

    return raw, data, primary


def categorical_replace(data, column, prefix, reference, levels=None):
    data = data.copy()
    levels = sorted(data[column].dropna().unique()) if levels is None else levels
    if reference not in levels:
        raise ValueError(f'Reference category {reference} is absent from {column}')
    terms = []
    for level in levels:
        if level == reference:
            continue
        term = f'{prefix}_{int(level)}'
        data[term] = (data[column] == level).astype(float)
        terms.append(term)
    metadata = {
        'source': column,
        'levels': list(levels),
        'reference': reference,
        'terms': terms,
    }
    return data, terms, metadata


def specifications(data, primary):
    """Return the two main models and the retained linear sensitivity models."""
    specs = []

    def add(name, frame, covariates=None, exposures=None, domains=None, metadata=None):
        specs.append({
            'name': name,
            'data': frame.copy(),
            'covariates': list(COVARIATES if covariates is None else covariates),
            'exposures': list(SUBJECTIVE if exposures is None else exposures),
            'domains': domains or {key: list(value) for key, value in DOMAINS.items()},
            'coding': metadata or {},
        })

    def pair(name, frame, covariates=None, metadata=None):
        add(name, frame, covariates=covariates, metadata=metadata)
        contextual_name = 'contextual' if name == 'subjective' else name + '_contextual'
        add(
            contextual_name,
            frame,
            covariates=covariates,
            exposures=SUBJECTIVE + OBJECTIVE,
            metadata=metadata,
        )

    pair('subjective', primary)
    pair(
        'range226',
        data[data.HasExam & data.RangeEligible],
        metadata={'change': 'Remove 12-month recency restriction only'},
    )

    lower_eye = primary.copy()
    lower_eye['VA'] = lower_eye.VA_Lower
    pair(
        'lower_eye',
        lower_eye,
        metadata={'change': 'Lower-eye continuous acuity; activity equations unchanged'},
    )

    age_frame, age_terms, age_meta = categorical_replace(primary, 'Age', 'Age', 3)
    pair(
        'age_categorical',
        age_frame,
        covariates=[value for value in COVARIATES if value != 'Age'] + age_terms,
        metadata=age_meta,
    )

    near_work = primary.copy()
    near_work['NearWorkGroup'] = np.minimum(near_work.NearWork, 3)
    near_work, near_terms, near_meta = categorical_replace(
        near_work, 'NearWorkGroup', 'NearWorkGroup', 1
    )
    pair(
        'nearwork_three',
        near_work,
        covariates=COVARIATES + near_terms,
        metadata=near_meta,
    )

    add('density_adjusted', primary, covariates=COVARIATES + ['DensityPop', 'DensityBuilt'])

    safety_exposures = [value for value in SUBJECTIVE if value not in DOMAINS['Safety']]
    safety_exposures.append('OverallSafety')
    safety_domains = {key: list(value) for key, value in DOMAINS.items()}
    safety_domains['Safety'] = ['OverallSafety']
    add('overall_safety', primary, exposures=safety_exposures, domains=safety_domains)

    park = primary.copy()
    park['ParkTimeGroup'] = np.select(
        [park.ParkTimeOriginal.isin([1, 2]), park.ParkTimeOriginal == 3,
         park.ParkTimeOriginal.isin([4, 5])],
        [1, 2, 3], default=np.nan,
    )
    park, park_terms, park_meta = categorical_replace(
        park, 'ParkTimeGroup', 'ParkTimeGroup', 3
    )
    park_exposures = [value for value in SUBJECTIVE if value != 'ParkAccess'] + park_terms
    park_domains = {key: list(value) for key, value in DOMAINS.items()}
    park_domains['Walking access'] = park_terms
    add(
        'park_access_three', park, exposures=park_exposures,
        domains=park_domains, metadata=park_meta,
    )

    income = primary.copy()
    income['IncomeGroup'] = np.select(
        [income.Income.isin([1, 2]), income.Income == 3, income.Income.isin([4, 5])],
        [1, 2, 3], default=np.nan,
    )
    income, income_terms, income_meta = categorical_replace(
        income, 'IncomeGroup', 'IncomeGroup', 1
    )
    add(
        'income_three',
        income,
        covariates=[value for value in COVARIATES if value != 'Income'] + income_terms,
        metadata=income_meta,
    )

    add(
        'recent204_boundary_audit',
        data[data.HasExam & data.RecentEligible],
        metadata={'scope': 'Boundary audit only; includes nine out-of-range records.'},
    )
    return specs
