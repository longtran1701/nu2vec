#!/bin/bash

SLURM_FILE=$1

while true
do
    tail -n 1 $SLURM_FILE 
    sleep 1
    clear
done
