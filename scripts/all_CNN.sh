#!/bin/bash
for i in $(seq 1)
do 
echo $i
sbatch -p bahram  -N 1 --time=72:00:00 /home/sshirzad/workspace/deepdrug/scripts/RunCNN.sh $((i-1))
done

