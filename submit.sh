#!/bin/bash
#SBATCH --job-name=data_loader_thermodrift
#SBATCH -p stf
#SBATCH -A stf
#SBATCH --nodes=1
#SBATCH --time=0:02:00
#SBATCH --ntasks=1
#SBATCH --mem=64G


source activate thermodrift
python -u test.py >> processing.log














