#!/bin/bash
#SBATCH --job-name=data_loader_thermodrift
#SBATCH -p stf
#SBATCH -A stf
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=75G
#SBATCH -o slurm/data_loader.out

source activate thermo_env
python -u Data_loader.py >> processing.log
