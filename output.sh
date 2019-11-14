#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --output=/home/magod/projects/def-lulam50/magod/vecmap/output/%x-%j.out
echo 'Hello, world!'
