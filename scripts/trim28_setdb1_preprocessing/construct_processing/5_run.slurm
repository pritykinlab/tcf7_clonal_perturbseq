#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64GB         # memory per cpu-core (4G is default)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=dillonlue@princeton.edu
#SBATCH -o 5_run.out

sh 5_run.sh
