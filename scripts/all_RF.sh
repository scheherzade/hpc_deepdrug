#!/bin/bash
for i in $(seq 2 100)
do 
test_id=$((i-1))
echo $test_id
qsub /home/sshirz1/runs/scripts/RunRF.sh -F $test_id -q workq -o "RF_${test_id}_o" -e "RF_${test_id}_e"
done

