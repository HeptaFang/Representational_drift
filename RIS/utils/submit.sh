bsub -q subscription -n 1 -R 'rusage[mem=4GB] span[hosts=1]' -a 'docker(continuumio/anaconda3)' -G 'compute-hiratani-t2' bash training.sh
