import argparse
import os

import yaml

from src.domain.table import get_table1, get_table2, get_table3, get_table4


def generate_table_1(configs):
    """
    This function generates the table 1 from the paper.
    """
    table = get_table1(configs)
    table.write(os.path.join(configs['output_path'], 'tables_and_plots/table1.tex'))


def generate_table_2(configs):
    """
    This function generates the table 2 from the paper.
    """
    table = get_table2(configs)
    table.write(os.path.join(configs['output_path'], 'tables_and_plots/table2.tex'))


def generate_table_3(configs):
    """
    This function generates the table 3 from the paper.
    """
    table = get_table3(configs)
    table.write(os.path.join(configs['output_path'], 'tables_and_plots/table3.tex'))


def generate_table_4(configs):
    """
    This function generates the table 4 from the paper.
    """
    table = get_table4(configs)
    table.write('./output/tables_and_plots/table4')


def main():
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    base_configs = argument_parser.parse_args()
    base_configs = vars(base_configs)
    generate_table_1(base_configs)
    generate_table_2(base_configs)
    generate_table_3(base_configs)
    generate_table_4(base_configs)


if __name__ == '__main__':
    main()
