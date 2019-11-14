#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --cpus-per-task=6
#SBATCH --mem=1G                  # memory (per node)
#SBATCH --time=0-00:05            # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca
#SBATCH --mail-type=FAIL
#SBATCH --output=~/projects/def-lulam50/magod/vecmap/outputs/slurm-x-%j.out

mkdir -p ~/projects/def-lulam50/magod/vecmap/outputs

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r beluga_requirements.txt

date
SECONDS=0

python test.py

diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
