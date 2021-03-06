#!/bin/bash
#SBATCH --job-name=thermodrift
#SBATCH -p ckpt
#SBATCH -A stf-ckpt
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH -o ./slurm_out/slurm_%j.out # STDOUT

script='/usr/lusers/jgershon/ThermoDrift/train_script.py'
main_dir='/gscratch/stf/jgershon/experiments'

#Name experiment here
experiment_dir='medium_width_drop15v3'

#Code to see if experiment name already exists
if [ -d "$main_dir/$experiment_dir" ]
then
	echo "Directory $experiemnt_dir already exists."
	exit 0
fi
echo "Making new directory for $experiment_dir"
mkdir "$main_dir/$experiment_dir"
mkdir "$main_dir/$experiment_dir/save_model"

indir='/gscratch/stf/jgershon/experiments/medium_width_drop15v2/save_model/model_500.pt'
outdir="$main_dir/$experiment_dir/"

autopep8 --in-place /usr/lusers/jgershon/ThermoDrift/train_script.py
autopep8 --in-place /usr/lusers/jgershon/ThermoDrift/thermodrift_model.py

source activate thermodrift
python -u $script -outdir $outdir -indir $indir >> $outdir/train.log














