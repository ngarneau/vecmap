import unittest

import yaml

from src.domain.table_generator.table import get_table1, get_table2


class TestTableGeneration(unittest.TestCase):
    CONFIGS = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)

    def test_table1_number_of_configs(self):
        """
        We have 1 experiment for 4 languages so 4 configs
        """
        table1 = get_table1(self.CONFIGS)
        configs = [c for _, e in table1.get_experiments() for c in e.get_parameters_combinations()]
        self.assertEqual(4, len(configs), msg="Table 1 should have 4 configurations")

    def test_table2_number_of_configs(self):
        """
        We have 8 experiment (including original) for 4 languages so 32 configs
        """
        table1 = get_table2(self.CONFIGS)
        configs = [c for _, e in table1.get_experiments() for c in e.get_parameters_combinations()]
        self.assertEqual(32, len(configs), msg="Table 2 should have 32 configurations")


if __name__ == '__main__':
    unittest.main()
