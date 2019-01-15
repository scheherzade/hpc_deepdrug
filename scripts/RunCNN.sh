#!/bin/bash
python_script='CNN_12_14.py'
date_str='2018-11-13-1221'
curr_date=`date '+%Y-%m-%d-%H%M'`
dir="/home/sshirzad/workspace/deepdrug"
results_dir="$dir/results/$date_str/CNN/$1"
mkdir -p ${results_dir}
mkdir -p "${results_dir}/npy"

cd ${results_dir}
ipython $dir/python_scripts/${python_script} $1 "$results_dir/npy" | tee ${results_dir}
#cp $dir/python_scripts/${python_script} ${results_dir}

