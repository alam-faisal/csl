#!/bin/bash -l
#SBATCH --job-name=job_30gs
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --mem=256G
#SBATCH --no-requeue
#SBATCH --output=/vast/home/alam/csl/jobs/job_30gs.out
#SBATCH --error=/vast/home/alam/csl/jobs/job_30gs.err
#SBATCH --qos=long
conda activate qaravan-env
python /vast/home/alam/csl/jobs/30gs_run.py
