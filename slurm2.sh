#!/bin/bash
#SBATCH --job-name=RESUME_TAGGER
#SBATCH --output=RESUME_TAGGER_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nimishbajaj@ufl.edu
#SBATCH --nodes=8              # Number of nodes
#SBATCH --ntasks=8             # Number of MPI ranks
#SBATCH --ntasks-per-node=1    # Number of MPI ranks per node
#SBATCH --ntasks-per-socket=1  # Number of tasks per processor socket on the node
#SBATCH --cpus-per-task=8      # Number of OpenMP threads for each MPI process/rank
#SBATCH --mem-per-cpu=4000mb   # Per processor memory request
#SBATCH --time=4-00:00:00      # Walltime in hh:mm:ss or d-hh:mm:ss
date;hostname;pwd

module load intel/2018 openmpi/3.1.0 tensorflow
pip install nltk

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /home/nimishbajaj/resume_tagger/resume_classifier.py

date
