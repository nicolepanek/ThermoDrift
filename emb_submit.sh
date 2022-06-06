#!/bin/bash

script='/home/ec2-user/ThermoDrift/emb_train.py'
main_dir='/home/ec2-user/ThermoDrift/experiments'

#Name experiment here
experiment_dir='trainv5'

#Code to see if experiment name already exists
if [ -d "$main_dir/$experiment_dir" ]
then
	echo "Directory $experiemnt_dir already exists."
	exit 0
fi
echo "Making new directory for $experiment_dir"
mkdir "$main_dir/$experiment_dir"
mkdir "$main_dir/$experiment_dir/save_model"

#indir='/gscratch/stf/jgershon/experiments/medium_widthv6/save_model/model_2000.pt'
outdir="$main_dir/$experiment_dir/"
data_dir="/home/ec2-user/ThermoDrift/data/"

#autopep8 --in-place /home/ec2-user/ThermoDrift/emb_train.py
#autopep8 --in-place /usr/lusers/jgershon/ThermoDrift/thermodrift_mo.py

source activate pytorch

python -u $script -data_dir $data_dir -outdir $outdir >> $outdir/train.log














