import argparse
import logging
import os

import mlflow
import yaml

from src.domain.table_generator.table import get_table1, get_table2
from src.scripts.runner import default_launcher, Launcher, configure_logging

DEFAULT_LOCAL_MLFLOW_OUTPUT = 'mlruns'


def main(args):
    run_launcher = default_launcher
    mlflow.set_tracking_uri(DEFAULT_LOCAL_MLFLOW_OUTPUT)
    num_runs = args['num_runs']
    cuda = args['cuda']

    launcher = Launcher(run_launcher, cuda)

    logging.info("Lauching experiments for Table 1")
    table1 = get_table1(args)
    launcher.run_experiment_for_table(table1)
    table1.write(os.path.join(args['output_path'], 'tables_and_plots/table1.tex'))
    logging.info("Done.")

    logging.info("Lauching experiments for Table 2")
    table2 = get_table2(args)
    launcher.run_experiment_for_table(table2)
    table2.write(os.path.join(args['output_path'], 'tables_and_plots/table2.tex'))
    logging.info("Done.")


if __name__ == '__main__':
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    base_configs = argument_parser.parse_args()
    base_configs = vars(base_configs)

    logging_path = os.path.join(base_configs['output_path'], 'logs')
    os.makedirs(logging_path, exist_ok=True)
    configure_logging(logging.INFO)

    main(base_configs)
