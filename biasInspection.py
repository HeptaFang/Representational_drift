import torch
import numpy as np
import os

weight = torch.load('model\\Artificial_MultiWithLatent_mul_0.0_-2.0_900.m')
print(weight['bias'])
print(torch.mean(weight['bias']), torch.std(weight['bias']))
