bsub -q subscription -n 32 -R 'rusage[mem=6GB] span[hosts=1]' -a 'docker(continuumio/anaconda3)' -G 'compute-hiratani-t2' bash batch_training.sh
