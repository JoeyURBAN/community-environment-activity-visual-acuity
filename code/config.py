from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'results'
INPUTS = {
    'survey': BASE / 'data/input_LOCAL_ONLY/geocoded_samples.xlsx',
    'unit_links': BASE / 'data/input_LOCAL_ONLY/sample_town_link_LOCAL_ONLY.csv',
    'unit_metrics': BASE / 'data/unit_environment_metrics.csv',
}

SEED = 20260908
N_BOOT = 5000
ALPHA = 0.05

# Internal name, retained survey column, questionnaire item, domain, display label.
ITEMS = [
    ('Walkability', 'built_walkability', 'Q37.3', 'Walking and traffic space', 'Crossing and pavement continuity'),
    ('Separation', 'built_separation', 'Q37.6', 'Walking and traffic space', 'Vehicle-pedestrian separation'),
    ('Greenery', 'built_green', 'Q37.4', 'Greenery', 'Perceived greenery'),
    ('PlaySpace', 'built_playground', 'Q37.5', 'Play space', 'Sufficiency of play space'),
    ('Shade', 'built_shade', 'Q37.7', 'Seasonal comfort', 'Summer shade'),
    ('WindComfort', 'built_wind', 'Q37.8', 'Seasonal comfort', 'Winter wind comfort'),
    ('TrafficSafety', 'safety_traffic', 'Q32.2', 'Safety', 'Traffic safety'),
    ('FacilitySafety', 'safety_facility', 'Q32.3', 'Safety', 'Facility and activity-site safety'),
    ('PersonalSafety', 'safety_security', 'Q32.4', 'Safety', 'Personal security'),
    ('ParkAccess', 'walk2park_min', 'Q38', 'Walking access', 'Shorter walking-time category'),
    ('AirQuality', 'air_pollution_visible', 'Q39', 'Pollution and disturbance', 'Fewer perceived air-pollution sources'),
    ('LightComfort', 'light_pollution_visible', 'Q41', 'Pollution and disturbance', 'Less disturbing night lighting'),
    ('Quietness', 'noise_level', 'Q43', 'Pollution and disturbance', 'Perceived quietness'),
]

SUBJECTIVE = [row[0] for row in ITEMS]
COVARIATES = ['Age', 'Female', 'ParentOne', 'ParentTwo', 'Income', 'Education']
# R2 main contextual model; the two alternatives are supplementary comparisons.
OBJECTIVE = ['Vegetation', 'Road', 'Facility', 'Building', 'ParkShare']
OBJECTIVE_JOINT = ['Vegetation', 'NDVI', 'Road', 'Facility', 'Building', 'ParkShare']
OBJECTIVE_NDVI = ['NDVI', 'Road', 'Facility', 'Building', 'ParkShare']
# Load and scale all six source measures once, using the same 195-child reference.
ALL_OBJECTIVE = list(OBJECTIVE_JOINT)
DOMAINS = {domain: [row[0] for row in ITEMS if row[3] == domain]
           for domain in dict.fromkeys(row[3] for row in ITEMS)}
