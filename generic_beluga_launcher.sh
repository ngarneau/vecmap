#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --array=1-10
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=5G                  # memory (per node)
#SBATCH --time=0-01:30            # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/magod/vecmap/slurm_outputs/%j.out

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r beluga_requirements.txt

date
SECONDS=0

python -m src.scripts.main_loop --seed=$SLURM_ARRAY_TASK_ID $@

diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date