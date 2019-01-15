#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=72:00:00
#PBS -m abe
#PBS -M sshirz1@lsu.edu

python_script='randomforest_args.py'
date_str='2018-11-13-1221'
curr_date=`date '+%Y-%m-%d-%H%M'`
dir="/home/sshirz1/runs"
results_dir="$dir/results/$date_str/RF/$1"
mkdir -p ${results_dir}
rm -rf "${results_dir}/*"
mkdir -p "${results_dir}/npy"
touch ${results_dir}/results_RF_${date_str}_$1.txt

cd ${results_dir}
ipython $dir/python_scripts/${python_script} $1 "$results_dir/npy" | tee ${results_dir}/results_RF_${date_str}_$1.txt

#cp $dir/python_scripts/${python_script} ${results_dir}

