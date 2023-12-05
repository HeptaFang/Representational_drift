GLOBAL_PATH = '.'
# GLOBAL_PATH = '/data/data_TT'

# Artificial dataset
BIN_NUM = 32
SESSION_NUM = 32
HIDDEN_NUM = 32
MODEL_HIDDEN_NUM = 32
CELL_NUM = 128
SPARSENESS = 0.1
FIT_ORDER = 7

# training parameters
NOISE_LEVELS = (0.0, 0.5, 1.0, 5.0)
BIAS_LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)
# NOISE_LEVELS = (0.0, 0.5, 1.0)
# BIAS_LEVELS = (-2.0, -1.0)
N_REPEAT = 8

# preprocessing parameters
SELECTED = False
POOLED = True
