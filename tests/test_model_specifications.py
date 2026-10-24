"""Configuration regression checks using synthetic, non-participant records."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import numpy as np
import pandas as pd
from config import ALL_OBJECTIVE, COVARIATES, OBJECTIVE, OBJECTIVE_JOINT, OBJECTIVE_NDVI, SUBJECTIVE
from data_preparation import specifications
from run_analysis import compatible_result


class ModelSpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 195
        cls.frame = pd.DataFrame({
            key: np.arange(n, dtype=float) % 5 + 1
            for key in SUBJECTIVE + COVARIATES + ALL_OBJECTIVE
        })
        cls.frame['group'] = np.arange(n) % 90
        cls.frame['Age'] = np.arange(n) % 4 + 3
        cls.frame['NearWork'] = np.arange(n) % 4 + 1
        cls.frame['ParkTimeOriginal'] = np.arange(n) % 5 + 1
        cls.frame['VA'] = 5.0
        cls.frame['VA_Lower'] = 4.9
        for name in ['HasExam', 'RangeEligible', 'RecentEligible']:
            cls.frame[name] = True
        cls.default = {s['name']: s for s in specifications(cls.frame, cls.frame)}
        cls.all_specs = {s['name']: s for s in specifications(cls.frame, cls.frame, True)}

    def test_main_contextual_has_five_indicators(self):
        self.assertEqual(OBJECTIVE, ['Vegetation', 'Road', 'Facility', 'Building', 'ParkShare'])
        self.assertEqual(self.default['contextual']['exposures'], SUBJECTIVE + OBJECTIVE)
        self.assertEqual(self.default['subjective']['exposures'], SUBJECTIVE)

    def test_four_contextual_sensitivities_use_same_five(self):
        for name in ['range226_contextual', 'lower_eye_contextual',
                     'age_categorical_contextual', 'nearwork_three_contextual']:
            self.assertEqual(self.default[name]['exposures'], SUBJECTIVE + OBJECTIVE)

    def test_alternatives_are_opt_in_and_preserve_order(self):
        self.assertEqual(len(self.default), 15)
        self.assertEqual(len(self.all_specs), 17)
        self.assertNotIn('joint_original', self.default)
        self.assertNotIn('ndvi_only', self.default)
        self.assertEqual(self.all_specs['joint_original']['exposures'], SUBJECTIVE + OBJECTIVE_JOINT)
        self.assertEqual(self.all_specs['ndvi_only']['exposures'], SUBJECTIVE + OBJECTIVE_NDVI)

    def test_alternatives_preserve_sample_covariates_and_domains(self):
        main = self.default['contextual']
        for name in ['joint_original', 'ndvi_only']:
            alternative = self.all_specs[name]
            pd.testing.assert_frame_equal(alternative['data'], main['data'])
            self.assertEqual(alternative['covariates'], main['covariates'])
            self.assertEqual(alternative['domains'], main['domains'])

    def test_multiplicity_family_sizes_follow_exposure_count(self):
        for name, expected in [('contextual', (38, 18, 36)),
                               ('joint_original', (40, 19, 38)),
                               ('ndvi_only', (38, 18, 36))]:
            spec = self.all_specs[name]
            p = len(spec['exposures'])
            self.assertEqual((2*p + 2, p, 2*p), expected)
            self.assertEqual(2*len(spec['domains']), 14)

    def test_resume_rejects_old_six_indicator_result(self):
        spec = self.default['contextual']
        saved = {'name': 'contextual', 'n': 195, 'clusters': 90,
                 'exposures': list(spec['exposures']), 'covariates': list(COVARIATES),
                 'coding': dict(spec['coding'])}
        self.assertTrue(compatible_result(saved, spec))
        old = copy.deepcopy(saved)
        old['exposures'] = SUBJECTIVE + OBJECTIVE_JOINT
        self.assertFalse(compatible_result(old, spec))
        changed = copy.deepcopy(saved)
        changed['n'] = 194
        self.assertFalse(compatible_result(changed, spec))


if __name__ == '__main__':
    unittest.main()
