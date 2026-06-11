#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32GB         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=dillonlue@princeton.edu
#SBATCH -o run.out

snakemake -c1 all \
    --latency-wait 60 \
    -j 2000 --nolock --cluster-config cluster.json --rerun-incomplete --keep-going \
    --cluster "sbatch --cpus-per-task {cluster.cpus-per-task} -n {cluster.ntasks}  -t {cluster.time} --mem {cluster.mem} -o logs/%j.out" \
    --jobname "snakejob.{jobid}.sh"
