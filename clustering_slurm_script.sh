#!/bin/bash

#SBATCH --job-name='CRAB_EM_Run'
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=education-tpm-msc-epa

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip
module load py-tqdm

pip install --user --upgrade "ema_workbench==2.5.1"
#pip install --user -e git+https://github.com/quaquel/EMAworkbench@mpi_update#egg=ema-workbench

mpiexec -n 1 python analaysis/timeseriesclustering.py
