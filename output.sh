#!/bin/bash
#SBATCH --account=def-lulam50
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
echo 'Hello, world!'