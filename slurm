#!/bin/bash
#SBATCH --job-name=RESUME_TAGGER# Job name
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nimishbajaj@ufl.edu # Where to send mail
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=2gb # Job memory request
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log # Standard output and error log
#SBATCH --partition=gpu #SBATCH --gpus=2
#SBATCH --output=/home/nimishbajaj/logs/resumeTagger_%j.out
#SBATCH --error=/home/nimishbajaj/logs/resumeTagger_%j.err

pwd; hostname;
date
module load python echo "Running plot script on a single CPU core"
python /home/nimishbajaj/resume_tagger/resume_classifier.py
date