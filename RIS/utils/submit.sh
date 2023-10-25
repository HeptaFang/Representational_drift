bsub -q subscription -n 8 -R 'rusage[mem=4GB] span[hosts=1]' -a 'docker(continuumio/anaconda3)' -G 'compute-hiratani-t2' bash batch_training.sh
