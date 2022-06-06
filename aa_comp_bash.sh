#!/bin/bash

script='/home/ubuntu/ThermoDrift/train_script_w_aa_composition.py'
main_dir='/home/ubuntu/ThermoDrift/experiments'

#Name experiment here
experiment_dir='aa_compv1'

#Code to see if experiment name already exists
if [ -d "$main_dir/$experiment_dir" ]
then
	echo "Directory $experiemnt_dir already exists."
	exit 0
fi
echo "Making new directory for $experiment_dir"
mkdir "$main_dir/$experiment_dir"
mkdir "$main_dir/$experiment_dir/save_model"

#indir='/gscratch/stf/jgershon/experiments/aa_compv1/save_model/model_3000.pt'
outdir="$main_dir/$experiment_dir"
data_dir="/home/ubuntu/ThermoDrift/data_for_zip/"

autopep8 --in-place /home/ubuntu/ThermoDrift/train_script_w_aa_composition.py
autopep8 --in-place /home/ubuntu/ThermoDrift/thermodrift_model_seqfrac.py

source activate thermo_drift_env
python -u $script -data_dir $data_dir -outdir $outdir >> $outdir/train.log














