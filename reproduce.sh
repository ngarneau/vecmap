#!/bin/bash
#

dvc pull
python -m src.scripts.experiment with 'cuda='${CUDA}
