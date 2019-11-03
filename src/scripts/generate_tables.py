import yaml

from src.domain.table import get_table1, get_table2, get_table3


def generate_table_1(configs):
    """
    This method generates the table 1 from the paper.
    """
    table = get_table1(configs)
    table.write('./output/tables_and_plots/table1.tex')


def generate_table_2(configs):
    """
    This method generates the table 1 from the paper.
    """
    table = get_table2(configs)
    table.write('./output/tables_and_plots/table2.tex')


def generate_table_3(configs):
    """
    This method generates the table 1 from the paper.
    """
    table = get_table3(configs)
    table.write('./output/tables_and_plots/table3.tex')


def main():
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    # generate_table_1(base_configs)
    # generate_table_2(base_configs)
    generate_table_3(base_configs)

if __name__ == '__main__':
    main()
