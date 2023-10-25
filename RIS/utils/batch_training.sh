#!/bin/bash

source activate Torch

max_parallel=8
current_jobs=0

for a in {0..10}; do
    for b in {0..13}; do
        # Check if the maximum parallel limit has been reached
        while [ $current_jobs -ge $max_parallel ]; do
            sleep 1
            current_jobs=$(jobs | wc -l)
        done

        # Run the command in the background
        python main.py $a $b &
        current_jobs=$(jobs | wc -l)
    done
done

# Wait for all background jobs to finish
wait
