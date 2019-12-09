#!/bin/bash

P=$1
Q=$2
R=$3

source activate ns_final
cd ../../../../src/
python man.py --input data.txt --keep coexpression cooccurence \
       experimental --p $P --q $Q --r $R

