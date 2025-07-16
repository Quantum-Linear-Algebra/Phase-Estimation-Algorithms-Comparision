#!/bin/bash
#SBATCH --job-name hardware_two_sites
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH --output logs/batch.out
#SBATCH --error logs/batch.err

cd $PSCRATCH
cd Phase-Estimation-Algorithms-Comparision
source ../Qiskit/bin/activate
python Comparision.py