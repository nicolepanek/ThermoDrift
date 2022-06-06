#!/bin/bash
#SBATCH --job-name=Analysis2_aa_comp_feature
#SBATCH -p ckpt
#SBATCH -A stf-ckpt
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH -o /usr/lusers/aultl/slurm_out/slurm_%j.out

set -x # put every command after into slurm_out job log 

script='/usr/lusers/aultl/ThermoDrift/model_eval/analysis_larissa_aa_comp.py'


autopep8 --in-place /usr/lusers/aultl/ThermoDrift/model_eval/analysis_larissa_aa_comp.py
autopep8 --in-place /usr/lusers/aultl/ThermoDrift/model_eval/inference_script.py


source activate /usr/lusers/aultl/anaconda3/envs/thermo_drift_env
python -u $script >> output/job.log














