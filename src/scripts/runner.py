import argparse
import logging
import os
import subprocess
import sys

import mlflow
import yaml

from src.domain.table_generator.table import get_table1, get_table2, get_table3, get_grid_search_experiments
from src.scripts.main_loop import run_main

DEFAULT_LOCAL_EMBEDDING_OUTPUT = 'output'
DEFAULT_LOCAL_MLFLOW_OUTPUT = 'mlruns'


def args_formatter(run_args):
    return ['--{}={}'.format(name, value) for name, value in run_args.items() if name != 'normalize']


def default_launcher(run_args, num_runs, cuda):
    run_args['num_runs'] = num_runs
    run_args['cuda'] = cuda
    run_args['embedding_output_uri'] = DEFAULT_LOCAL_EMBEDDING_OUTPUT
    run_args['mlflow_output_uri'] = DEFAULT_LOCAL_MLFLOW_OUTPUT
    for run_number in range(num_runs):
        run_args['seed'] = run_number
        run_main(run_args)


def configure_logging(log_level):
    """
    Configure logger

    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


class Launcher:
    def __init__(self, run_launcher, cuda):
        self.run_launcher = run_launcher
        self.cuda = cuda

    def run_experiment_for_table(self, table):
        for name, experiment in table.get_experiments():
            logging.info("Running experiment: {}".format(experiment.EXPERIMENT_NAME))
            for config in experiment.get_parameters_combinations():
                config['experiment_name'] = experiment.EXPERIMENT_NAME
                if 'cuda' in experiment.CHANGING_PARAMS:
                    self.run_launcher(config, experiment.NUM_RUNS, cuda=config['cuda'])
                else:
                    self.run_launcher(config, experiment.NUM_RUNS, cuda=self.cuda)
                logging.info("Done running experiment: {} with override {}".format(experiment.EXPERIMENT_NAME, config))


def main(args):
    run_launcher = default_launcher
    mlflow.set_tracking_uri(DEFAULT_LOCAL_MLFLOW_OUTPUT)

    cuda = args['cuda']

    launcher = Launcher(run_launcher, cuda)

    logging.info("Lauching experiments for Table 1")
    table1 = get_table1(args)
    launcher.run_experiment_for_table(table1)
    logging.info("Done.")

    logging.info("Lauching experiments for Table 2")
    table2 = get_table2(args)
    launcher.run_experiment_for_table(table2)
    logging.info("Done.")

    logging.info("Lauching experiments for Table 3")
    table3 = get_table3(args)
    launcher.run_experiment_for_table(table3)
    logging.info("Done.")

    logging.info("Lauching experiments for grid search experiments")
    table4 = get_grid_search_experiments(base_configs)
    launcher.run_experiment_for_table(table4)
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
