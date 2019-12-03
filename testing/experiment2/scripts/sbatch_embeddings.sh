#!/bin/bash

SBATCH_OPTS="\
--mail-type=END --mail-user=Henri.Schmidt@tufts.edu \
-N 1 -n 1 --mem=32000 --time=7-00:00:00 \
"

VALUES=(0.25 0.5 1 2 4)

for P in "${VALUES[@]}"
do
    for Q in "${VALUES[@]}"
    do
        for R in  "${VALUES[@]}"
        do
            sbatch $SBATCH_OPTS ./embedding.sh $P $Q $R
        done
    done
done

