import os

from train_artificial_dataset import main as train_artificial_dataset


def main():
    noise_level = 0.0
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:
            for seed in range(N_REPEAT):
                main(noise_level, bias, seed)
