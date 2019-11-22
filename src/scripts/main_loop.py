# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import logging
import os
import sys

import mlflow
import yaml

from src.domain.vecmap import VecMap
from src.handler.mlflow_handler import get_mlflow_logging_handler
from src.validations import whitening_arguments_validation


def run_experiment(_config):
    logging.info(_config)
    mlflow.log_params(_config)

    whitening_arguments_validation(_config)

    vec_map = VecMap(_config)

    vec_map.load_embeddings()

    vec_map.embeddings_normalization_step()

    vec_map.allocate_memory()

    vec_map.fully_unsupervised_initialization_step()

    vec_map.robust_self_learning()

    vec_map.eval()


def run_main(configs):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    path_to_log_directory = os.path.join(configs['output_path'], 'logs')
    mlflow_logging_handler = get_mlflow_logging_handler(path_to_log_directory, logging.INFO, formatter)
    logger.addHandler(mlflow_logging_handler)

    mlflow.set_tracking_uri(configs['mlflow_output_uri'])
    mlflow.set_experiment(configs['experiment_name'])  # Create the experiment if it did not already existed

    with mlflow.start_run():
        try:
            run_experiment(configs)
        except KeyboardInterrupt:
            logging.warning("Run exited.")

    logger.removeHandler(mlflow_logging_handler)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument('--{}'.format(config),
                                         type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                                         default=value)
        else:
            argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    options = argument_parser.parse_args()
    configs = vars(options)
    run_main(configs)
