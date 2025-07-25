#!/bin/bash
#SBATCH --job-name two_site_sim
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH --nodes 1
#SBATCH --output logs/sim2site.out
#SBATCH --error logs/sim2site.err

source ../Qiskit-env/bin/activate
python Comparison.py