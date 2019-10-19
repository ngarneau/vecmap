#!/bin/bash
#

dvc pull
python -m src.scripts.experiment $@
