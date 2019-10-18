#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=6G                  # memory (per node)
#SBATCH --time=0-05:00            # time (DD-HH:MM)
#SBATCH --output=$HOME/vecmap/run_outputs/%j.out

source $HOME/venv/bin/activate
python -m src.scripts.experiment -u with seed_dictionary_method='unsupervised' cuda=True num_runs=1