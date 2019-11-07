import datetime
import logging
import os

import mlflow


class MlFlowHandler(logging.FileHandler):
    def emit(self, record):
        super(MlFlowHandler, self).emit(record)
        mlflow.log_artifact(self.baseFilename)


def get_mlflow_logging_handler(path_to_log_directory, log_level, formatter):
    log_filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.log'
    fh = MlFlowHandler(filename=os.path.join(path_to_log_directory, log_filename))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    return fh
