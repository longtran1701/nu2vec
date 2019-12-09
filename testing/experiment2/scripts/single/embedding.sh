#!/bin/bash

P=$1
Q=$2

source activate ns_final
cd ../../../../src/
python man.py --input data.txt --keep experimental --p $P --q $Q

