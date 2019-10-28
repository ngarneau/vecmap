import yaml

from src.domain.table import get_table1


def generate_table_1(configs):
    """
    This method generates the table 1 from the paper.
    """
    table = get_table1(configs)
    table.write('./output/tables_and_plots/table1.tex')


def main():
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    generate_table_1(base_configs)

if __name__ == '__main__':
    main()
