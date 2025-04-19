import os

script_names = ["sp30_10.py", "sp30_20.py", "sp30_30.py"]#, "sp30_40.py", "sp30_50.py", "sp30_60.py"]  
job_names = ["sp30_10", "sp30_20", "sp30_30"]#, "sp30_40", "sp30_50", "sp30_60"]

for script_name, job_name in zip(script_names, job_names):

    JOBS_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of sbatcher.py
    SCRIPT_PATH = os.path.join(JOBS_DIR, script_name)

    command = f"python {SCRIPT_PATH}"

    def darwin_sbatch(job_name, command, run_dir, data_dir):
        sbatch_file = os.path.join(run_dir, f"{job_name}.sbatch")
        log_file = os.path.join(data_dir, f"{job_name}")

        with open(sbatch_file, "w") as f:
            f.write("#!/bin/bash -l\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --exclusive\n")
            f.write("#SBATCH --time=2-0:00:00\n")
            f.write("#SBATCH --mem=256G\n")
            f.write("#SBATCH --no-requeue\n")
            f.write(f"#SBATCH --output={log_file}.out\n")
            f.write(f"#SBATCH --error={log_file}.err\n")
            f.write("#SBATCH --qos=long\n")
            f.write("conda activate qaravan-env\n")
            f.write(command + "\n")

        os.system(f"sbatch {sbatch_file}")

    darwin_sbatch(job_name, command, JOBS_DIR, JOBS_DIR)
    print(f"Sbatch job '{job_name}' submitted.")
