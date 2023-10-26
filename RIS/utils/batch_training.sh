#!/bin/bash

source activate Torch

max_parallel=32
current_jobs=0

for a in {0..7}; do
    for b in {0..9}; do
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
