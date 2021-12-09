#!/bin/bash
#BATCH --job-name=nimishbajaj    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nimishbajaj@ufl.edu     # Where to send mail	
#SBATCH --ntasks=16                    # Run on a single CPU
#SBATCH --mem=48gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=nimishbajaje_%j.log   # Standard output and error log
pwd; hostname; date

module load tensorflow
pip install nltk

echo "Running resume tagger on a single multi core"

python /home/nimishbajaj/resume_tagger/resume_classifier.py

date
