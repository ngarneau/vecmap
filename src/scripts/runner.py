import logging
import argparse
import subprocess
import mlflow
from copy import deepcopy
import sys
import yaml

from src.scripts.experiment import run_main
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


def main(args):
    if args.supercomputer:
        run_launcher = supercomputer_launcher
        mlflow.set_tracking_uri(DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT)
    else:
        run_launcher = default_launcher
        mlflow.set_tracking_uri(DEFAULT_LOCAL_MLFLOW_OUTPUT)

    num_runs = args.num_runs
    cuda = args.cuda

    # Reproduce original run
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)

    # Run table1 experiments
    table1 = get_table1()
    for name, experiment_cls in table1.get_experiments():
        experiment = experiment_cls(base_configs)
        mlflow.set_experiment(experiment.EXPERIMENT_NAME)
        for config in experiment.get_parameters_combinations():
            run_launcher(config, num_runs, cuda)

    # Run table2 experiments
    table2 = get_table2()
    for name, experiment_cls in table2.get_experiments():
        experiment = experiment_cls(base_configs)
        mlflow.set_experiment(experiment.EXPERIMENT_NAME)
        for config in experiment.get_parameters_combinations():
            if 'vocabulary_cutoff' in experiment.EXPERIMENT_NAME:
                run_launcher(config, num_runs, cuda=False)
            else:
                run_launcher(config, num_runs, cuda)


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
