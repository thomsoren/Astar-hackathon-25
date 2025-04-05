#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=MuZero
#SBATCH --time=0-00:30:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:h100
#SBATCH --mem=8G                 # Asking for xGB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 # x CPU cores
#SBATCH --output=idun-log.txt      # Specifying ‘stdout’
#SBATCH --error=idun-log.err        # Specifying ‘stderr’
#SBATCH --mail-user=chrisfjo@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Python/3.11.5-GCCcore-13.2.0
pip install --upgrade pip
cd DELIVERY
cd PROBLEM1
pip install -r ./requirements.txt --user
pip list

python --version
python main.py