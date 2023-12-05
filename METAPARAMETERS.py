PARAMETER_MODE = 'manual'
# PARAMETER_MODE = 'json'

if PARAMETER_MODE == 'json':
    import json
    # read parameters from a file
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    GLOBAL_PATH = parameters['GLOBAL_PATH']

    # Artificial dataset
    BIN_NUM = parameters['BIN_NUM']
    SESSION_NUM = parameters['SESSION_NUM']
    HIDDEN_NUM = parameters['HIDDEN_NUM']
    MODEL_HIDDEN_NUM = parameters['MODEL_HIDDEN_NUM']
    CELL_NUM = parameters['CELL_NUM']
    SPARSENESS = parameters['SPARSENESS']
    FIT_ORDER = parameters['FIT_ORDER']

    # training parameters
    NOISE_LEVELS = parameters['NOISE_LEVELS']
    BIAS_LEVELS = parameters['BIAS_LEVELS']
    N_REPEAT = parameters['N_REPEAT']

    # preprocessing parameters
    SELECTED = parameters['SELECTED']
    POOLED = parameters['POOLED']


elif PARAMETER_MODE == 'manual':
    GLOBAL_PATH = '.'
    # GLOBAL_PATH = '/data/data_TT'

    # Artificial dataset
    BIN_NUM = 32
    SESSION_NUM = 32
    HIDDEN_NUM = 32
    MODEL_HIDDEN_NUM = 32
    CELL_NUM = 128
    SPARSENESS = 0.1
    FIT_ORDER = 3

    # training parameters
    NOISE_LEVELS = (0.0, 0.5)
    BIAS_LEVELS = (-2.0, -1.0)
    N_REPEAT = 8

    # preprocessing parameters
    SELECTED = False
    POOLED = True
