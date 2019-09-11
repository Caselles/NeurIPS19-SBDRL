from constants import *
import numpy as np
import torch
import os
from tqdm import tqdm


all_subdirs = ['vae/'+d for d in os.listdir('vae/') if os.path.isdir('vae/'+d)]

latest_subdir = max(all_subdirs, key=os.path.getmtime)

vae = torch.load(latest_subdir+'/saved_models/epoch_10_env_0', map_location={'cuda:0': 'cpu'})

actions = np.load('../learn_4_dim_linear_disentangled_representation/actions.npy')

obs = torch.Tensor(np.load('../learn_4_dim_linear_disentangled_representation/inputs.npy').reshape(-1,64,64,3).transpose((0,3,1,2)))

states = []

for elem in tqdm(obs):

    states.append(vae.forward(elem.reshape(1,3,64,64), encode=True, mean=True).detach().numpy())

np.save('../downstream_tasks/states_method_non-linear-AE', np.array(states).reshape(-1,Z_DIM))
