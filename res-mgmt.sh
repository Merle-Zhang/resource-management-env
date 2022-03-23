#!/bin/bash

#SBATCH --job-name res-mgmt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 00:10:00
#SBATCH --partition veryshort
#SBATCH --output res-mgmt.out

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

#! Run the executable

~/.conda/envs/res-mgmt-rl-dev/bin/python3 random_ran_env.py
