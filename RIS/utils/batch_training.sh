#!/bin/bash

source activate Torch

max_parallel=32
current_jobs=0

for a in {0..3}; do
    for b in {0..4}; do
        for c in {0..7}; do
            # Check if the maximum parallel limit has been reached
            while [ $current_jobs -ge $max_parallel ]; do
                sleep 1
                current_jobs=$(jobs | wc -l)
            done

            # Run the command in the background
            python main.py $a $b $c &
            current_jobs=$(jobs | wc -l)
        done
    done
done

# Wait for all background jobs to finish
wait
