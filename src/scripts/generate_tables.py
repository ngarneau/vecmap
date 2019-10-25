from src.domain.table import Table1


def generate_table_1():
    """
    This method generates the table 1 from the paper.
    """
    experiment_name = "main_table"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    table = Table1(experiment_name, runs)
    table.write('./output/tables_and_plots')


def main():
    generate_table_1()

if __name__ == '__main__':
    main()
