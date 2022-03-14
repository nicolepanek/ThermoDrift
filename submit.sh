#!/bin/bash
#SBATCH --job-name=thermodrift
#SBATCH -p stf
#SBATCH -A stf
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH -o ./slurm_out/slurm_%j.out # STDOUT

name='increaseing_width.log'

source activate thermodrift
python -u train_script.py >> $name














