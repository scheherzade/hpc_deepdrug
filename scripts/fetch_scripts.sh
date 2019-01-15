#!/bin/bash
RF_file1="https://raw.githubusercontent.com/scheherzade/deepdrug/master/scripts/all_RF.sh"
RF_file2="https://raw.githubusercontent.com/scheherzade/deepdrug/master/scripts/RunRF.sh"
CNN_file1="https://raw.githubusercontent.com/scheherzade/deepdrug/master/scripts/all_CNN.sh"
CNN_file2="https://raw.githubusercontent.com/scheherzade/deepdrug/master/scripts/RunCNN.sh"
if [ $1 == "RF" ]
then 
rm all_RF.sh
rm RunRF.sh
wget $RF_file1
wget $RF_file2
elif [ $1 == "CNN" ]
then
rm all_CNN.sh
rm RunCNN.sh
wget $CNN_file1
wget $CNN_file2
elif [ $1 == "all" ]
then
rm all_RF.sh
rm RunRF.sh
rm all_CNN.sh
rm RunCNN.sh

wget $CNN_file1
wget $CNN_file2
wget $RF_file1
wget $RF_file2
fi
chmod +x all_RF.sh 
chmod +x RunRF.sh 

