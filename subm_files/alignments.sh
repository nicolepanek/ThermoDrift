#!/bin/bash
#SBATCH --job-name=combined_alignments_thermodrift
#SBATCH -p ckpt
#SBATCH -A stf-ckpt
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH -o /usr/lusers/achazing/thermo_proteins/ThermoDrift/slurm_out/combined_alignments.out

source activate /usr/lusers/achazing/anaconda3/envs/thermo_env
python -u /usr/lusers/achazing/thermo_proteins/ThermoDrift/alignments/fasta_seq_similarity.py -seq /usr/lusers/achazing/thermo_proteins/ThermoDrift/data/combined.fasta -dist_cutoff 5 >> align_processing.log
