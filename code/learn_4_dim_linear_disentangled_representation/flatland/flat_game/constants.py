import torch
import math

#savefolder = '/media/looka/bc6982c9-5dc3-4761-93f3-40a4eafda3ec/phd/flatland/WorldModels/mazes/VAE_data/'
#savefolder_idea = '/media/looka/bc6982c9-5dc3-4761-93f3-40a4eafda3ec/phd/flatland/WorldModels/mazes/VAE_data_idea/'
savefolder = '/media/looka/bc6982c9-5dc3-4761-93f3-40a4eafda3ec/phd/flatland/WorldModels/mazes/VAE_data_idea/'

##### CONFIG

torch.set_printoptions(precision=10)
## CUDA variable from Torch
CUDA = torch.cuda.is_available()
#torch.backends.cudnn.deterministic = True
## Dtype of the tensors depending on CUDA
GPU_MODE = True
DEVICE = torch.device("cuda") if GPU_MODE else torch.device("cpu")

#Display
DISPLAY = False


## Eps for log
EPSILON = 1e-6

## VAE
LATENT_VEC = 64
MAX_BATCH = 19
INPUT_DIM = (64,64,3)
CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']
DENSE_SIZE = 1024
CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']
Z_DIM = 2
NUMBER_OF_EPOCHS_VAE = 11
BATCH_SIZE_VAE = 128
LAMBDA_RECONSTRUCTION = 1
NB_EPISODES_VIZU = 10
BETA = 1
ANNEAL = .995
THRESHOLD_EARLY_STOPPING = 0.0001


## RNN
OFFSET = 1
HIDDEN_UNITS = 2048
HIDDEN_DIM = 2048
TEMPERATURE = 1
GAUSSIANS = 8
NUM_LAYERS = 1
SEQUENCE = 499
PARAMS_FC1 = HIDDEN_UNITS * NUM_LAYERS * 2
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)
SEQUENCE_ROLLOUT = 1
ROLLOUT_LENGTH = 20 
FF_FC1 = 1024
FF_FC2 = 1024
FF_FC3 = 1024
BATCH_SIZE_FF = 32


## Images
HEIGHT = 64
WIDTH = 64

## Training
ADAM = True
LR = 1e-3
L2_REG = 1e-4
BATCH_SIZE_LSTM = 1
NUMBER_OF_EPOCHS_LSTM = 2000

## Refresh
LOG_PERIOD = 5
VALIDATE_PERIOD = 1
SAVING_PERIOD = 1
FIG_EARLY_STOP_PERIOD = 1
STATS_PERIOD = 1

## Env
ENV_NAME = 'mazes'
BATCH_SIZE_ROLLOUTS = 50
NUMBER_OF_TIMESTEPS = 1000
NB_OF_ENVS = 3
NB_OF_ENVS_IDEA = 2

## Env for validation
ENV_NAME = 'mazes'
BATCH_SIZE_ROLLOUTS_V = 5
NUMBER_OF_TIMESTEPS_V = 500
NB_OF_ENVS_V = 3
NB_OF_ENVS_V_IDEA = 3

## CMA-ES
ACTION_SPACE = 3
#SAVEFOLDER = '/media/looka/bc6982c9-5dc3-4761-93f3-40a4eafda3ec/phd/flatland/WorldModels/flatland_topview/data/'
VAE_PATH = '/home/looka/workspace/phd/shallow_mind_lab/WorldModels/flatland_topview_wordmodels/vae/15307154025863242_torch_good/saved_models/epoch_10_vae'
LSTM_PATH = '/home/looka/workspace/phd/shallow_mind_lab/WorldModels/flatland_topview_wordmodels/mdn_rnn/15308770915195956_torch_good_fixed_hidden_state_init/saved_models/epoch_40_mdn_rnn'
DAE_PATH = '/home/looka/workspace/phd/shallow_mind_lab/WorldModels/ICLR2019_1D/dae/1536245524011803/saved_models/epoch_10_dae'
CONTROLLER_PATH = None
SIGMA_INIT = 0.5
POPULATION = 152
PARALLEL = 4
N_GENERATIONS = 2000


INPUT_SHAPE_FLATTEN = 192


# GENERATIVE REPLAY
NB_SAMPLES_GR = 1000000
SAMPLING_STRAT = 'naive'
CUSTOM_SCALE = 1
FOLDER_VAE = 'vae/15427054160891235_ft_0_1/'

# AUTOMATIC DETECTION OF ENVIRONMENT CHANGE
P_VALUE_THRESHOLD = 0.01
SIZE_TEST = 100

# FORWARD MODEL FF PARAMETERS
BATCH_SIZE_FF = 128
NB_ACTIONS = 4
EMBEDDING_ACTION_SIZE = 1
NUMBER_OF_EPOCHS_FF = 1000
