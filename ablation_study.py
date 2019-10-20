import argparse
import subprocess
import mlflow
from copy import deepcopy

DEFAULT_SUPERCOMPUTER_EMBEDDING_OUTPUT = '/project/def-lulam50/magod/vecmap/output'
DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT = 'file:/project/def-lulam50/magod/vecmap/mlflow'
EXPERIMENT_NAME = 'ablation_study'

default_params = {
    'stochastic_initial': 0.1,
    'vocabulary_cutoff': 20000,
    'csls': 10,
    'direction': 'union',
    'reweight': 0.5,
    'seed_dictionary_method': 'unsupervised'
}

ablation_dict = {
    'stochastic_initial': [1.0],
    'vocabulary_cutoff': [0],
    'csls': [0],
    'direction': ['forward'],
    'reweight': [1.0],
    'seed_dictionary_method': ['random_raw', 'random_cutoff']
}


def run_args_formatter(run_args):
    return ['--{}={}'.format(name, value) for name, value in run_args.items()]


def supercomputer_launcher(run_args, num_runs, cuda):
    print('cuda is ', cuda)
    run_args['supercomputer'] = True
    run_args['num_runs'] = 1
    run_args['cuda'] = cuda
    run_args['embedding_output_uri'] = DEFAULT_SUPERCOMPUTER_EMBEDDING_OUTPUT
    run_args['mlflow_output_uri'] = DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT
    run_args['exp_name'] = EXPERIMENT_NAME

    for run_number in range(num_runs):
        run_args['seed'] = run_number

        subprocess.Popen(['sbatch', 'generic_beluga_launcher.sh', *run_args_formatter(run_args)])


def default_launcher(run_args, num_runs, cuda):
    # TODO: Sequential launcher
    pass


def main(args):
    if args.supercomputer:
        run_launcher = supercomputer_launcher
        mlflow.set_tracking_uri(DEFAULT_SUPERCOMPUTER_MLFLOW_OUTPUT)
    else:
        raise ValueError('Only supercomputer mode is currently supported.')

    mlflow.set_experiment(EXPERIMENT_NAME)
    num_runs = args.num_runs
    cuda = args.cuda

    # Reproduce original run
    run_launcher(default_params, num_runs, cuda)

    # Ablated runs
    for ablated_param, param_values in ablation_dict.items():
        for param_value in param_values:
            run_params = deepcopy(default_params)
            run_params[ablated_param] = param_value

            if ablated_param == 'vocabulary_cutoff':
                print("Stepped into the if")
                run_launcher(run_params, num_runs, cuda=False)
            else:
                run_launcher(run_params, num_runs, cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=10, help='The number of runs to execute per configuration.')
    parser.add_argument('--supercomputer',
                        action='store_true',
                        help='Wether or not the ablation study has to be parallelized on a supercomputer.')
    parser.add_argument('--cuda', action='store_true', help='Wether or not to use a GPU to run the ablation study.')
    args = parser.parse_args()
    main(args)