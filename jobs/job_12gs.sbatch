#!/bin/bash -l
#SBATCH --job-name=job_12gs
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --mem=256G
#SBATCH --no-requeue
#SBATCH --output=/vast/home/alam/csl/jobs/job_12gs.out
#SBATCH --error=/vast/home/alam/csl/jobs/job_12gs.err
#SBATCH --qos=long
conda activate qaravan-env
python /vast/home/alam/csl/jobs/12gs_run.py
