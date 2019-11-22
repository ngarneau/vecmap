from collections import defaultdict
from itertools import product

import numpy as np
from mlflow.tracking import MlflowClient


class Experiment:
    EXPERIMENT_NAME = 'None'
    FIXED_PARAMS = {}
    LANGUAGE_PARAMS = {}
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        self.base_config = base_config
        self.mlflow_client = MlflowClient(tracking_uri=base_config['mlflow_output_uri'])

    def get_runs(self):
        mlflow_experiment = self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME)
        return self.mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id])

    def get_parameters_combinations(self):
        run_params = self.base_config
        run_params.update(self.FIXED_PARAMS)
        for source_language in self.LANGUAGE_PARAMS['source_language']:
            for target_language in self.LANGUAGE_PARAMS['target_language']:
                if source_language != target_language:
                    run_params['source_language'] = source_language
                    run_params['target_language'] = target_language
                    if len(self.CHANGING_PARAMS) > 0:
                        param_values = product(*self.CHANGING_PARAMS.values())
                        ablated_params = self.CHANGING_PARAMS.keys()
                        for params_combination in param_values:
                            # Apply the language params combination
                            new_params = {
                                ablated_param: param_value
                                for ablated_param, param_value in zip(ablated_params, params_combination)
                            }
                            run_params.update(new_params)
                            yield run_params, self.get_sbatch_args(run_params)
                    else:
                        yield run_params, self.get_sbatch_args(run_params)

    def get_sbatch_args(self, run_params):
        return {}


class OriginalExperiment(Experiment):
    EXPERIMENT_NAME = 'original'
    FIXED_PARAMS = {
        'stochastic_initial': 0.1,
        'vocabulary_cutoff': 20000,
        'csls': 10,
        'direction': 'union',
        'reweight': 0.5,
        'seed_dictionary_method': 'unsupervised'
    }
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['de', 'es', 'fi', 'it']}
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        super().__init__(base_config)

    def _is_a_valid_run(self, run):
        return (run.data.params['target_language'] in self.LANGUAGE_PARAMS['target_language']
                and run.data.params['source_language'] in self.LANGUAGE_PARAMS['source_language']
                and run.info.status == 'FINISHED')

    def aggregate_runs(self):
        runs = self.get_runs()
        accuracies = defaultdict(list)
        times = defaultdict(list)
        for run in runs:
            if self._is_a_valid_run(run):
                minutes = ((run.info.end_time - run.info.start_time) // 60 // 60) % 60
                accuracies[run.data.params['target_language']].append(run.data.metrics['accuracy'] * 100)
                times[run.data.params['target_language']].append(minutes)
        return {'accuracies': accuracies, 'times': times}


class StochasticAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'stochastic_ablation'
    CHANGING_PARAMS = {
        'stochastic_initial': [1.0],
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class VocabularyCutOffAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'vocabulary_cutoff_ablation'
    CHANGING_PARAMS = {'vocabulary_cutoff': [10 ** 5], 'cuda': [False]}

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'cpus-per-task': 20, 'mem': '30G', 'time': '7-0:00', 'gres': 'gpu:0'}


class CSLSAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'csls_ablation'
    CHANGING_PARAMS = {
        'csls': [0],
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class DirectionAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'direction_ablation'
    CHANGING_PARAMS = {
        'direction': ['forward'],
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class ReweightAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'reweight_ablation'
    CHANGING_PARAMS = {
        'reweight': [1.0],
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class RandomSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'random_seed_dictionary_ablation'
    CHANGING_PARAMS = {'seed_dictionary_method': ['random_raw']}

    def __init__(self, base_config):
        super().__init__(base_config)


class RandomCutoffSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'random_cutoff_seed_dictionary_ablation'
    CHANGING_PARAMS = {'seed_dictionary_method': ['random_cutoff']}

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesOriginalExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'other_languages_original'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['et', 'fa', 'lv', 'vi']}

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-1:00'}


class OtherLanguagesStochasticAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'other_languages_stochastic_ablation'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['et', 'fa', 'lv', 'vi']}
    CHANGING_PARAMS = {
        'stochastic_initial': [1.0],
    }

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-1:00'}


class OtherLanguagesStochasticExperiment(StochasticAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_stochastic'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesCSLSAblationExperiment(CSLSAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_csls_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesDirectionAblationExperiment(DirectionAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_direction_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesReweightAblationExperiment(ReweightAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_reweight_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesRandomSeedDictionaryAblationExperiment(RandomSeedDictionaryAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_random_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesRandomCutoffSeedDictionaryAblationExperiment(RandomCutoffSeedDictionaryAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_random_cutoff_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-1:00'}


class OtherLanguagesRandomSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'other_languages_random_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['et', 'fa', 'lv', 'vi']}
    CHANGING_PARAMS = {'seed_dictionary_method': ['random_raw']}

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-1:00'}


class OtherLanguagesRandomCutoffSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'other_languages_random_cutoff_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['et', 'fa', 'lv', 'vi']}
    CHANGING_PARAMS = {'seed_dictionary_method': ['random_cutoff']}

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-1:00'}


class GridSearchExperiment(OriginalExperiment):
    def __init__(self, base_config):
        super().__init__(base_config)

    def __get_run_params(self, run):
        if len(self.CHANGING_PARAMS) == 1:
            return run.data.params[list(self.CHANGING_PARAMS)[0]]
        else:
            return tuple(run.data.params[param_name] for param_name in self.CHANGING_PARAMS)

    def aggregate_runs(self):
        runs = self.get_runs()
        accuracies = defaultdict(lambda: defaultdict(list))
        times = defaultdict(lambda: defaultdict(list))
        num_iters = defaultdict(lambda: defaultdict(list))
        avg_iter_times = defaultdict(lambda: defaultdict(list))
        for run in runs:
            if self._is_a_valid_run(run):
                minutes = max(((run.info.end_time - run.info.start_time) // 60 // 60) % 60, 1)
                run_params = self.__get_run_params(run)
                accuracies[run.data.params['target_language']][run_params].append(run.data.metrics['accuracy'] * 100)
                times[run.data.params['target_language']][run_params].append(minutes)

                if 'num_iters' in run.data.metrics:
                    num_iters[run.data.params['target_language']][run_params].append(int(run.data.metrics['num_iters']))

                # If part of runs without the iter_duration logging, we approximate
                # an iteration duration with (total_duration/num_iter)
                if 'iter_duration' in run.data.metrics:
                    avg_iter_times[run.data.params['target_language']][run_params].append(
                        np.mean(run.data.metrics['iter_duration']))
                else:
                    avg_iter_times[run.data.params['target_language']][run_params].append(minutes * 60 /
                                                                                          run.data.metrics['num_iters'])

        return {'accuracies': accuracies, 'times': times, 'num_iters': num_iters, 'avg_iter_times': avg_iter_times}


class CSLSGridSearchExperiment(GridSearchExperiment):
    EXPERIMENT_NAME = 'csls_grid_search'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['de', 'es', 'fi', 'it']}

    CHANGING_PARAMS = {'csls': list(range(1, 21))}

    def __init__(self, base_config):
        super().__init__(base_config)


class VocabularyCutoffGridSearchExperiment(GridSearchExperiment):
    EXPERIMENT_NAME = 'vocabulary_cutoff_grid_search'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['de', 'es', 'fi', 'it']}

    CHANGING_PARAMS = {'vocabulary_cutoff': [int(factor * 10 ** 3) for factor in range(10, 31)]}

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-2:00'}


class StochasticGridSearchExperiment(GridSearchExperiment):
    EXPERIMENT_NAME = 'stochastic_grid_search'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['de', 'es', 'fi', 'it']}

    CHANGING_PARAMS = {
        'stochastic_initial': list(np.linspace(0.05, 0.3, 5)),
        'stochastic_multiplier': list(np.linspace(1.5, 3, 4))
    }

    def __init__(self, base_config):
        super().__init__(base_config)

    def get_sbatch_args(self, run_params):
        return {'time': '0-2:30'}


class HyperparametersExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'hyperparameters'
    LANGUAGE_PARAMS = {'source_language': ['en'], 'target_language': ['de', 'es', 'et', 'fa', 'fi', 'it' 'lv', 'vi']}

    # TODO Change a bunch of params here
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        super().__init__(base_config)
