#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=5G                  # memory (per node)
#SBATCH --time=0-05:00            # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca
#SBATCH --mail-type=FAIL

source $HOME/venv/bin/activate
python -m src.scripts.experiment $@