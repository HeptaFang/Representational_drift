bsub -q general -n 1 -R 'rusage[mem=4GB] span[hosts=1]' -a 'docker(continuumio/anaconda3)' -G 'compute-hiratani' bash training.sh
