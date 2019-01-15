#!/bin/bash
RF_file="https://raw.githubusercontent.com/scheherzade/deepdrug/master/python_scripts/randomforest_args.py"
CNN_file="https://raw.githubusercontent.com/scheherzade/deepdrug/master/python_scripts/CNN_12_14.py"
if [ $1 == "RF" ]
then 
rm randomforest_args.py
wget $RF_file
elif [ $1 == "CNN" ]
then
rm CNN_12_14.py
wget $CNN_file
elif [ $1 == "all" ]
then
rm randomforest_args.py
rm CNN_12_14.py
wget $CNN_file
wget $RF_file
fi

