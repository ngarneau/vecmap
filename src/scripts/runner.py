import logging
import argparse
import subprocess
import mlflow
from copy import deepcopy
import sys
import yaml

from src.scripts.main_loop import run_main
from src.domain.table import get_table1, get_table2

DEFAULT_SUPERCOMPUTER_EMBEDDING_OUTPUT = '/scratch/magod/vecmap/output'
DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT = 'file:/scratch/magod/vecmap/mlflow'
DEFAULT_LOCAL_EMBEDDING_OUTPUT = 'output'
DEFAULT_LOCAL_MLFLOW_OUTPUT = 'mlruns'
EXPERIMENT_NAME = 'ablation_study'


def run_args_formatter(run_args):
    return ['--{}={}'.format(name, value) for name, value in run_args.items()]


def supercomputer_launcher(run_args, num_runs, cuda):
    run_args['supercomputer'] = True
    run_args['num_runs'] = 1
    run_args['cuda'] = cuda
    run_args['embedding_output_uri'] = DEFAULT_SUPERCOMPUTER_EMBEDDING_OUTPUT
    run_args['mlflow_output_uri'] = DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT
    run_args['experiment_name'] = EXPERIMENT_NAME

    for run_number in range(num_runs):
        run_args['seed'] = run_number
        subprocess.Popen(['sbatch', 'generic_beluga_launcher.sh', *run_args_formatter(run_args)])


def default_launcher(run_args, num_runs, cuda):
    run_args['num_runs'] = num_runs
    run_args['cuda'] = cuda
    run_args['embedding_output_uri'] = DEFAULT_LOCAL_EMBEDDING_OUTPUT
    run_args['mlflow_output_uri'] = DEFAULT_LOCAL_MLFLOW_OUTPUT
    run_args['experiment_name'] = EXPERIMENT_NAME
    run_main(run_args)




class Launcher:
    def __init__(self, run_launcher, num_runs, cuda):
        self.run_launcher = run_launcher
        self.num_runs = num_runs
        self.cuda = cuda

    def run_experiment_for_table(self, table):
        for name, experiment in table.get_experiments():
            mlflow.set_experiment(experiment.EXPERIMENT_NAME)
            logging.info("Running experiment: {}".format(experiment.EXPERIMENT_NAME))
            for config in experiment.get_parameters_combinations():
                if 'vocabulary_cutoff' in experiment.EXPERIMENT_NAME:
                    self.run_launcher(config, self.num_runs, cuda=False)
                else:
                    self.run_launcher(config, self.num_runs, self.cuda)
                logging.info("Done running experiment: {} with override {}".format(experiment.EXPERIMENT_NAME, config))


def main(args):
    if args.supercomputer:
        run_launcher = supercomputer_launcher
        mlflow.set_tracking_uri(DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT)
    else:
        run_launcher = default_launcher
        mlflow.set_tracking_uri(DEFAULT_LOCAL_MLFLOW_OUTPUT)

    num_runs = args.num_runs
    cuda = args.cuda
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)

    launcher = Launcher(run_launcher, num_runs, cuda)

    # Run table1 experiments
    table1 = get_table1(base_configs)
    launcher.run_experiment_for_table(table1)

    table2 = get_table2(base_configs)
    launcher.run_experiment_for_table(table2)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=10, help='The number of runs to execute per configuration.')
    parser.add_argument('--supercomputer',
                        action='store_true',
                        help='Wether or not the ablation study has to be parallelized on a supercomputer.')
    parser.add_argument('--cuda', action='store_true', help='Wether or not to use a GPU to run the ablation study.')
    args = parser.parse_args()
    main(args)
